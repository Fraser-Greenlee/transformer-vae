import torch
from torch import nn
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
import time

from datasets.features import ClassLabel
from transformers import trainer as trainer_script
from transformers.integrations import WandbCallback, is_wandb_available, TensorBoardCallback, CometCallback, AzureMLCallback, MLflowCallback
from transformer_vae.sequence_checks import SEQ_CHECKS
from transformer_vae.trainer_callback import WandbCallbackUseModelLogs


logger = logging.getLogger(__name__)


if WandbCallback in trainer_script.DEFAULT_CALLBACKS:
    # Allow tracking extra training losses via the model's `get_latest_logs` method
    trainer_script.DEFAULT_CALLBACKS.remove(WandbCallback)
    trainer_script.DEFAULT_CALLBACKS.append(WandbCallbackUseModelLogs)
    import wandb
else:
    logger.warn("Not using Weights and Biasis, this will give you incomplete logs.")


NOT_ALLOWED_LOGGERS = [TensorBoardCallback, CometCallback, AzureMLCallback, MLflowCallback]

for logger_integration in NOT_ALLOWED_LOGGERS:
    removed = []
    if logger_integration in trainer_script.DEFAULT_CALLBACKS:
        trainer_script.DEFAULT_CALLBACKS.remove(logger_integration)
        removed.append(logger_integration)
    logger.info(f"Only supports W&B logging, removed loggers: {removed}")


class VAE_Trainer(trainer_script.Trainer):
    def _text_from_latent(self, latent):
        # TODO can I do many latents in parallel?
        # TODO this may not work for Funnel-VAE
        generation = self.model.generate(latent=latent, bos_token_id=0, min_length=self.args.generate_min_len, max_length=self.args.generate_max_len)
        return self.tokenizer.decode(
            generation[0].tolist()
        )

    def _interpolate_samples(self, eval_dataset):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        mini_eval_dataloader_iter = iter(
            DataLoader(
                eval_dataset,
                sampler=RandomSampler(eval_dataset),
                batch_size=2,
                collate_fn=self.data_collator,
            )
        )

        samples = self._prepare_inputs(next(mini_eval_dataloader_iter))
        latents = self.model(**samples).latnet
        start_latent, end_latent = latents[0].view(1, -1), latents[1].view(1, -1)
        latent_diff = end_latent - start_latent

        table = wandb.Table(columns=["Interpolation Ratio", "Text"])
        table.add_data(-10, self.tokenizer.decode(samples['input_ids'][0]))
        for i in range(11):
            ratio = i / 10
            latent = start_latent + ratio * latent_diff
            table.add_data(ratio, self._text_from_latent(latent))
        table.add_data(10, self.tokenizer.decode(samples['input_ids'][1]))
        wandb.log({"interpolate points": table}, step=self.state.global_step)

    def _random_samples(self):
        table = wandb.Table(columns=["Text", "Valid", "IsMaxLen"])
        latent_points = torch.randn(self.args.n_random_samples, self.model.config.latent_size, device=self.model.device)
        # TODO can I greedy decode these in parallel?
        for i in range(latent_points.size(0)):
            text = self._text_from_latent(latent_points[i].view(1, -1))
            valid = None if not self.args.seq_check else SEQ_CHECKS[self.args.seq_check](text)
            table.add_data(text, valid, len(text) == self.args.generate_max_len)
        wandb.log({"random points": table}, step=self.state.global_step)

    def _clustering(self, eval_dataset, class_column_name):
        if class_column_name is None:
            for key, val in eval_dataset.features['category_num'].items():
                if type(val) is ClassLabel:
                    class_column_name = key
        # TODO plot t-SNE with points coloured for each category

    def evaluate(self, eval_dataset: Optional[Dataset] = None, class_column_name=None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        Adds extra VAE tests:
        - Interpolation between samples in latent space.
        - Random latent codes from normal distribution.
        if class column provided?
        - tSNE plots with class-label colouring.
        """
        if is_wandb_available():
            start_eval = time.time()
            with torch.no_grad():
                self.model.eval()
                self._interpolate_samples(eval_dataset)
                self._random_samples()
                # TODO add t-SNE clustering with class labels
            generate_time = time.time() - start_eval
        output_metrics = super().evaluate(eval_dataset=eval_dataset)
        if is_wandb_available():
            self.log({'eval_get_test_loss_time': time.time() - start_eval + generate_time})
            self.log({'eval_generate_time': generate_time})
        return output_metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # FIX ISSUE https://github.com/huggingface/transformers/issues/9057
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16 and trainer_script._use_native_amp:
                with trainer_script.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if isinstance(outputs, dict):
                    loss = outputs["loss"].mean().detach()
                    logits = (outputs.get('logits', None),)
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = (outputs.get('logits', None),)
                else:
                    logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = trainer_script.nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = trainer_script.nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)
