import time
import logging
import collections
from typing import Optional, Dict, List, Tuple, Union, Any
from tqdm import tqdm
import torch
from torch import nn, autograd
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast

from transformers import trainer as trainer_script
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
)
from transformers.file_utils import is_apex_available
if is_apex_available():
    from apex import amp

from transformer_vae.sequence_checks import SEQ_CHECKS
from transformer_vae.trainer_callback import WandbCallbackUseModelLogs
from transformer_vae.sklearn import train_svm
from transformer_vae.utils import slerp, SortishSampler


logger = logging.getLogger(__name__)


if WandbCallback in trainer_script.DEFAULT_CALLBACKS:
    # Allow tracking extra training losses via the model's `get_latest_logs` method
    trainer_script.DEFAULT_CALLBACKS.remove(WandbCallback)  # type: ignore
    trainer_script.DEFAULT_CALLBACKS.append(WandbCallbackUseModelLogs)  # type: ignore
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
    def __init__(self, model=None, args=None, **kwargs):
        self.latent_stack = torch.zeros(
            args.interpolate_training_step_rate * args.train_batch_size, model.config.latent_size,
            dtype=torch.float, device=args.device
        )
        self.final_decoder_hidden_state_stack = torch.zeros(
            args.interpolate_training_step_rate * args.train_batch_size, model.config.transformer.n_positions, model.config.transformer.d_model,
            dtype=torch.float, device=args.device
        )
        super().__init__(model, args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None
        src_lens = []
        for row in tqdm(self.train_dataset, desc='Calculating dataset item lengths.'):
            src_lens.append(len(row['input_ids']) - row['input_ids'].count(0))
        return SortishSampler(src_lens)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _tokens_from_latent(self, latent):
        with torch.no_grad():
            old = self.model.config.use_extra_logs
            self.model.config.use_extra_logs = False
            result = self.model.generate(
                input_ids=self.model.decoder_start_token_id * torch.ones((latent.size(0), 1), dtype=torch.long, device=self.args.device),
                latent=latent, bos_token_id=self.model.decoder_start_token_id, min_length=self.args.generate_min_len,
                max_length=self.args.generate_max_len
            )
            self.model.config.use_extra_logs = old
            return result

    def _text_from_latent(self, latent):
        return self.tokenizer.batch_decode(self._tokens_from_latent(latent), skip_special_tokens=True)

    def _interpolate_samples(self, eval_dataset):
        '''
            Interpolates between 2 latent encodings of real points.
            Results are logged to Weights and Biasis.
        '''
        mini_eval_dataloader_iter = iter(
            DataLoader(
                eval_dataset,
                sampler=RandomSampler(eval_dataset),
                batch_size=2,
                collate_fn=self.data_collator,
            )
        )
        samples = self._prepare_inputs(next(mini_eval_dataloader_iter))
        latents = self.model(**samples).latent
        interp_latent, interp_ratio = self.gradual_interpolation_inputs(latents[0], latents[1])

        seq_check_results = 0
        seq_check = SEQ_CHECKS[self.args.seq_check]
        table = wandb.Table(columns=["Interpolation Ratio", "Text", "Valid"])
        table.add_data(-10, self.tokenizer.decode(samples["input_ids"][0]), True)
        texts = self._text_from_latent(interp_latent)

        for i in range(11):
            valid = seq_check(texts[i])
            table.add_data(interp_ratio[i].item(), texts[i], valid)
            if i > 0 and i < 10:
                seq_check_results += int(valid)
        table.add_data(10, self.tokenizer.decode(samples["input_ids"][1]), True)

        wandb.log({"interpolate points": table}, step=self.state.global_step)
        if self.args.seq_check:
            wandb.log(
                {'interpolation samples passing seq check': seq_check_results / 9},
                step=self.state.global_step
            )

    def _random_samples(self):
        # TODO This should be random samples from the models prior but in an MMD-VAE the prior doesn't actually match a gaussian so this needs to change.
        table = wandb.Table(columns=["Text", "Valid"])
        seq_check_results = 0
        seq_check = SEQ_CHECKS[self.args.seq_check]
        latent_points = torch.randn(25, self.model.config.latent_size, device=self.model.device)
        texts = self._text_from_latent(latent_points)
        for txt in texts:
            valid = seq_check(txt)
            table.add_data(txt, valid)
            seq_check_results += int(valid)

        wandb.log({"random points": table}, step=self.state.global_step)
        if self.args.seq_check:
            wandb.log(
                {'random samples passing seq check': seq_check_results / latent_points.size(0)},
                step=self.state.global_step
            )

    def _latent_with_class(self, eval_dataset):
        dataloader = self.get_eval_dataloader(eval_dataset)
        latents_with_class = []
        for inputs in dataloader:
            if self.args.test_classification:
                class_label = inputs.pop("class_label")

            inputs = self._prepare_inputs(inputs)
            outputs = self.model(**inputs)

            row = [outputs.get("latent").tolist()]
            if self.args.test_classification:
                row.append(class_label.tolist())  # type: ignore

            latents_with_class.append(row)
        return latents_with_class

    def _svm_classification(self, latents_with_class):
        accuracy_log = train_svm(latents_with_class)
        wandb.log(accuracy_log, step=self.state.global_step)

    def _t_sne(self, latents_with_class):
        # TODO use wandb.plot
        """sample code
        table = wandb.Table(columns=["x", "y", "class"])
        points = t_sne(latents_with_class)
        for point in points:
            table.add_data(point.x, point.y, point.class)
        wandb.log({"my_custom_id" : wandb.plot.scatter(table, "x", "y", "class")}, step=x)
        """
        pass

    def _evaluate_latent_samples(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.args.sample_from_latent:
            self._interpolate_samples(eval_dataset)
            # self._random_samples()  Not using
        if self.args.test_classification:
            latents_with_class = self._latent_with_class(eval_dataset)
            self._svm_classification(latents_with_class)
        # self._t_sne(latents_with_class)

    def get_loss(self, outputs, labels):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            return self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def get_loss_grad(self, outputs, labels):
        if self.use_amp:
            with autocast():
                loss = self.get_loss(outputs, labels)
        else:
            loss = self.get_loss(outputs, labels)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss

    def gradual_interpolation_inputs(self, latent_start, latent_end):
        ratios = torch.arange(0, 1.1, 0.1, device=self.args.device)
        interpolations = slerp(ratios, latent_start.repeat(11, 1), latent_end.repeat(11, 1))
        return interpolations, ratios

    def random_interpolation_inputs(self, latent):
        batch_size = latent.size(0)
        interpolation_ratios = torch.rand(batch_size, device=self.args.device) * 0.5
        interpolation_ratios.requires_grad = True
        shifted_indices = torch.arange(latent.size(0), device='cpu')[1:].tolist() + [0]
        latent_interpolated = slerp(interpolation_ratios, latent, latent[shifted_indices])
        return latent_interpolated, interpolation_ratios

    def prepare_interpolation_data(self, latent, model):
        '''
        For optimising model interpolations directly, find interpolated latent codes with their ratio.
        Produces 1 interpolation for every 2 samples.
        '''
        original_latent = latent.detach()
        original_latent.requires_grad = True
        interp_latent, interp_ratio = self.random_interpolation_inputs(original_latent)
        tokens = self._tokens_from_latent(interp_latent)
        # don't log interpolation inference
        old = model.config.use_extra_logs
        model.config.use_extra_logs = False
        interp_outputs = model(decoder_input_ids=tokens, latent=interp_latent, output_hidden_states=True)
        model.config.use_extra_logs = old

        return interp_outputs.logits, interp_latent, interp_outputs.hidden_states[-1], interp_ratio

    def training_interpolation_step(self, final_decoder_hidden_states, latent, model):
        '''
            Only to be ran with Funnel-T5.
        '''
        interpolated_logits, interpolated_latent, interpolated_last_hidden_state, target_a = self.prepare_interpolation_data(latent, model)
        interpolated_last_hidden_state_d = interpolated_last_hidden_state.detach()

        if self.args.smooth_cosine or self.args.smooth_logits or self.args.smooth_logits_mean:
            # minimise cosine error between interpolated final decoder states
            if self.args.smooth_cosine:
                # low hidden units gradient w.r.t interpolation coeficient
                logits_wrt_alpha = autograd.grad(outputs=interpolated_last_hidden_state, inputs=target_a, only_inputs=True, create_graph=True, retain_graph=True)[0]
                smoothness_loss = logits_wrt_alpha.norm()
                smoothness_loss.backward(retain_graph=True)
            elif self.args.smooth_logits:
                # low logits gradient w.r.t interpolation coeficient
                logits_wrt_alpha = autograd.grad(outputs=interpolated_logits.norm(), inputs=target_a, only_inputs=True, create_graph=True, retain_graph=True)[0]
                smoothness_loss = logits_wrt_alpha.norm()
                smoothness_loss.backward(retain_graph=True)
            elif self.args.smooth_logits_mean:
                # low logits gradient w.r.t interpolation coeficient
                logits_wrt_alpha = autograd.grad(outputs=interpolated_logits.mean(), inputs=target_a, only_inputs=True, create_graph=True, retain_graph=True)[0]
                smoothness_loss = logits_wrt_alpha.mean()
            else:
                raise Exception('No smooth loss selected')
            smoothness_loss *= self.args.advisery_weight
            smoothness_loss.backward(retain_graph=True)
            model.latest_logs['smoothness_loss'] = model.latest_logs.get('smoothness_loss', 0) + smoothness_loss.item()

        if self.args.cycle_loss:
            # minimise cosine error between latent code & re-encoded latent code `latent VS Encode(Decode(latent))`
            target = 1.0 * torch.ones(interpolated_latent.size(0), device=self.args.device)
            old = model.config.use_extra_logs
            model.config.use_extra_logs = False
            cycle_loss = torch.nn.CosineEmbeddingLoss()(
                model(inputs_embeds=interpolated_last_hidden_state).latent, interpolated_latent, target
            )
            model.config.use_extra_logs = old
            cycle_loss *= self.args.cycle_weight
            cycle_loss.backward(retain_graph=True)
            model.latest_logs['cycle_loss'] = model.latest_logs.get('cycle_loss', 0) + cycle_loss.item()

        if model.critic:
            if self.state.global_step > self.args.min_critic_steps:
                # update model
                # accumulate compute graph on critic loss variable
                critic_loss_on_model = model.critic(interpolated_last_hidden_state).mean() * self.args.advisery_weight
                # get gradients of the output only w.r.t the inputs and not model.critic
                critic_loss_to_last_hidden = autograd.grad(outputs=critic_loss_on_model, inputs=interpolated_last_hidden_state, only_inputs=True, retain_graph=True)
                # acumulate gradient in VAE model (will only be the VAE-decoder)
                interpolated_last_hidden_state.backward(critic_loss_to_last_hidden, retain_graph=True)
                model.latest_logs['critic_loss_on_model'] = model.latest_logs.get('critic_loss_on_model', 0) + critic_loss_on_model.item()

            # update critic
            # real samples
            final_decoder_hidden_states.size(), latent.size()
            critic_loss = model.critic(final_decoder_hidden_states, torch.zeros((latent.size(0), 1), device=self.args.device)).mean()
            # interpolate samples
            interpolated_last_hidden_state_d = interpolated_last_hidden_state.detach()
            critic_loss += model.critic(interpolated_last_hidden_state_d, target_a.detach().view(-1, 1)).mean()
            # average between the 2 losses
            critic_loss /= 2
            critic_loss.backward(retain_graph=True)  # accumulate gradient on critic
            model.latest_logs['critic_loss'] = model.latest_logs.get('critic_loss', 0) + critic_loss.item()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Adds adviserial loss when using critic model.
        Adv is currently put on/off single GPU, will need to switch for multi-GPU training.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)

        if model.critic or (self.args.smooth_cosine or self.args.smooth_logits or self.args.smooth_logits_mean) or self.args.cycle_loss:
            pos = self.args.train_batch_size * (self.state.global_step % self.args.interpolate_training_step_rate)
            self.latent_stack[pos:pos + self.args.train_batch_size] = outputs.latent.detach()
            self.final_decoder_hidden_state_stack[pos:pos + self.args.train_batch_size] = outputs.decoder_hidden_states[-1].detach()
            if self.state.global_step > 0 and pos == 0:
                self.training_interpolation_step(self.final_decoder_hidden_state_stack, self.latent_stack, model)

        return self.get_loss_grad(outputs, labels).detach()

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        Adds extra VAE tests:
        - Interpolation between samples in latent space.
        - Random latent codes from normal distribution.
        if class column provided?
        - tSNE plots with class-label colouring.
        """
        if self.state.global_step < wandb.run.history._step:
            self.state.global_step = wandb.run.history._step
        if is_wandb_available():
            start_eval = time.time()
            with torch.no_grad():
                self.model.eval()
                self._evaluate_latent_samples(eval_dataset=eval_dataset)
            generate_time = time.time() - start_eval
        output_metrics = super().evaluate(eval_dataset=eval_dataset)
        if is_wandb_available():
            wandb.log({"eval_get_test_loss_time": time.time() - start_eval + generate_time}, step=self.state.global_step)  # type: ignore
            wandb.log({"eval_generate_time": generate_time}, step=self.state.global_step)  # type: ignore
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
            if self.use_amp:
                with trainer_script.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            if has_labels:
                if isinstance(outputs, dict):
                    loss = outputs["loss"].mean().detach()
                    logits = (outputs.get("logits", None),)
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = (outputs.get("logits", None),)
                else:
                    logits = outputs

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

        if logits is not None:
            logits = logits
        if labels is not None:
            labels = labels

        return (loss, logits, labels)
