import time
from typing import Optional, Dict, List
import wandb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader

from transformers import trainer as trainer_script
from transformers.utils import logging
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
)

from transformer_vae.sequence_checks import SEQ_CHECKS
from transformer_vae.trainer_callback import WandbCallbackUseModelLogs
from transformer_vae.sklearn import train_svm
from transformer_vae.utils import slerp

logger = logging.get_logger(__name__)


NOT_ALLOWED_LOGGERS = [TensorBoardCallback, CometCallback, AzureMLCallback, MLflowCallback]

for logger_integration in NOT_ALLOWED_LOGGERS:
    removed = []
    if logger_integration in trainer_script.DEFAULT_CALLBACKS:
        trainer_script.DEFAULT_CALLBACKS.remove(logger_integration)
        removed.append(logger_integration)
    logger.info(f"Only supports W&B logging, removed loggers: {removed}")


class VAE_Trainer(trainer_script.Trainer):
    text_to_array = None

    def __init__(self, model=None, args=None, custom_methods={}, **kwargs):
        self.clean_tkn_spaces = not args.dont_clean_up_tokenization_spaces
        if args.render_text_image:
            assert 'custom_text_to_array' in custom_methods
            self.text_to_array = custom_methods['custom_text_to_array']
        super().__init__(model, args, **kwargs)
        self.remove_callback(WandbCallback)
        self.add_callback(WandbCallbackUseModelLogs)

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
        return self.tokenizer.batch_decode(self._tokens_from_latent(latent), skip_special_tokens=True, clean_up_tokenization_spaces=self.clean_tkn_spaces)

    def _log_image(self, texts):
        '''
            Parse texts as images and log a single, long image to Weights and Biasis.
        '''
        single_image_array = np.concatenate([self.text_to_array(txt) * 255 for txt in texts], axis=1)
        wandb.log({"image_interpolation": [wandb.Image(single_image_array)]}, step=self.state.global_step)

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
        interp_latent, interp_ratio = self.gradual_interpolation_inputs(latents[0], latents[1], self.args.device, self.args.interpolate_all_at_once)
        start_txt = self.tokenizer.decode(samples["input_ids"][0], clean_up_tokenization_spaces=self.clean_tkn_spaces)
        end_txt = self.tokenizer.decode(samples["input_ids"][1], clean_up_tokenization_spaces=self.clean_tkn_spaces)
        texts = self._text_from_latent(interp_latent)

        if self.args.render_text_image:
            self._log_image([start_txt] + texts + [end_txt])

        seq_check_results = 0
        seq_check = SEQ_CHECKS[self.args.seq_check]
        table = wandb.Table(columns=["Interpolation Ratio", "Text", "Valid"])
        table.add_data(-10, start_txt, True)

        for i in range(11):
            valid = seq_check(texts[i], self.text_to_array)
            table.add_data(interp_ratio[i].item(), texts[i], valid)
            if i > 0 and i < 10:
                seq_check_results += int(valid)
        table.add_data(10, end_txt, True)

        wandb.log({"interpolate points": table}, step=self.state.global_step)
        if self.args.seq_check:
            wandb.log(
                {'interpolation samples passing seq check': seq_check_results / 9},
                step=self.state.global_step
            )

    def _latent_with_class(self, eval_dataset):
        dataloader = self.get_eval_dataloader(eval_dataset)
        latents_with_class = []
        for inputs in dataloader:
            class_label = inputs.pop("class_label")

            inputs = self._prepare_inputs(inputs)
            outputs = self.model(**inputs)

            latent = outputs.get("latent")
            latent = latent.reshape(latent.size(0), -1)  # join all latents into one
            row = [latent.tolist()]
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
        if self.args.test_classification:
            latents_with_class = self._latent_with_class(eval_dataset)
            self._svm_classification(latents_with_class)
        # self._t_sne(latents_with_class)

    @staticmethod
    def gradual_interpolation_inputs(latent_start, latent_end, device, interpolate_all_at_once):
        ratios = torch.arange(0, 1.1, 0.1, device=device)
        if len(latent_start.size()) > 2:
            # if using spectrum tokens just interpolate all at once
            latent_start = latent_start.view(latent_start.size(0), -1)
            latent_end = latent_start.view(latent_start.size(0), -1)
        num_latent_tokens = latent_start.size(0)
        latent_token_dim = latent_start.size(1)
        if interpolate_all_at_once:
            latent_start, latent_end = latent_start.view(1, -1), latent_end.view(1, -1)
            interpolations = slerp(ratios, latent_start.repeat(11, 1), latent_end.repeat(11, 1))
            return interpolations.view(11, num_latent_tokens, latent_token_dim), ratios
        # get list of repeating ratios so [0.0, 0.0, 0.1, 0.1,,,] when num_latent_tokens=2
        rep_ratios = ratios.repeat(num_latent_tokens, 1).T.reshape(-1)
        interpolations = slerp(rep_ratios, latent_start.repeat(11, 1), latent_end.repeat(11, 1))
        return interpolations.view(11, num_latent_tokens, latent_token_dim), ratios

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        Adds extra VAE tests:
        - Interpolation between samples in latent space.
        """
        if self.state.global_step < wandb.run.history._step:
            self.state.global_step = wandb.run.history._step
        if is_wandb_available():
            start_eval = time.time()
            with torch.no_grad():
                self.model.eval()
                self._evaluate_latent_samples(eval_dataset=eval_dataset)
            generate_time = time.time() - start_eval
        output_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if is_wandb_available():
            wandb.log({"eval_get_test_loss_time": time.time() - start_eval + generate_time}, step=self.state.global_step)  # type: ignore
            wandb.log({"eval_generate_time": generate_time}, step=self.state.global_step)  # type: ignore
        return output_metrics
