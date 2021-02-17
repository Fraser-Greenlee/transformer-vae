import time
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast

from transformers import trainer as trainer_script
from transformers.utils import logging
from transformers.optimization import AdamW, get_scheduler
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    is_fairscale_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
)
from transformers.file_utils import is_apex_available
if is_apex_available():
    from apex import amp

if is_fairscale_available():
    from fairscale.optim import OSS

from transformer_vae.optimizers import FixedAdafactor
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
        self.latent_stack = torch.zeros(
            args.interpolate_training_step_rate * args.train_batch_size, model.config.latent_size,
            dtype=torch.float, device=args.device
        )
        self.final_decoder_hidden_state_stack = torch.zeros(
            args.interpolate_training_step_rate * args.train_batch_size, model.config.t5.n_positions, model.config.t5.d_model,
            dtype=torch.float, device=args.device
        )
        self.clean_tkn_spaces = not args.dont_clean_up_tokenization_spaces
        if args.render_text_image:
            assert 'custom_text_to_array' in custom_methods
            self.text_to_array = custom_methods['custom_text_to_array']
        super().__init__(model, args, **kwargs)
        self.remove_callback(WandbCallback)
        self.add_callback(WandbCallbackUseModelLogs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Edited to use fixed Adafactor.

        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = FixedAdafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
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
        interp_latent, interp_ratio = self.gradual_interpolation_inputs(latents[0], latents[1])
        start_txt = self.tokenizer.decode(samples["labels"][0], clean_up_tokenization_spaces=self.clean_tkn_spaces)
        end_txt = self.tokenizer.decode(samples["labels"][1], clean_up_tokenization_spaces=self.clean_tkn_spaces)
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

    def _random_samples(self):
        raise NotImplementedError('Not sampling from true prioir here.')
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
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def gradual_interpolation_inputs(self, latent_start, latent_end):
        ratios = torch.arange(0, 1.1, 0.1, device=self.args.device)
        # handle slerp seperately for each latent token
        import pdb; pdb.set_trace()
        interpolations = slerp(ratios, latent_start.repeat(11, 1), latent_end.repeat(11, 1))
        return interpolations, ratios

    def random_interpolation_inputs(self, latent):
        batch_size = latent.size(0)
        interpolation_ratios = 0.5 - (torch.rand(batch_size, device=self.args.device) - 0.5).abs()
        latent_interpolated = slerp(interpolation_ratios, latent, latent[::-1])
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

        return interp_latent, interp_outputs.reconstructed_encoding, interp_outputs.hidden_states[-1], interp_ratio

    def training_interpolation_step(self, final_decoder_hidden_states, latent, model):
        '''
            Sample interpolations to add additional losses.

            None of these have substantially improved interpolation quality yet.
        '''
        interpolated_latent, reconstructed_encoding, interpolated_last_hidden_state, target_a = self.prepare_interpolation_data(latent, model)
        interpolated_last_hidden_state_d = interpolated_last_hidden_state.detach()

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
            cycle_loss /= interpolated_latent.size(0)
            cycle_loss.backward(retain_graph=True)
            model.latest_logs['cycle_loss'] = model.latest_logs.get('cycle_loss', 0) + cycle_loss.item()
        elif self.args.vae_cycle_loss:
            target = 1.0 * torch.ones(interpolated_latent.size(0), device=self.args.device)
            old = model.config.use_extra_logs
            model.config.use_extra_logs = False
            cycle_loss = torch.nn.CosineEmbeddingLoss()(
                model.vae(reconstructed_encoding, skip_reg_loss=True).latent, interpolated_latent, target
            )
            model.config.use_extra_logs = old
            cycle_loss *= self.args.cycle_weight
            cycle_loss /= interpolated_latent.size(0)
            cycle_loss.backward(retain_graph=True)
            model.latest_logs['cycle_loss'] = model.latest_logs.get('cycle_loss', 0) + cycle_loss.item()

        if model.critic:
            if self.state.global_step > self.args.min_critic_steps:
                # update model
                # accumulate compute graph on critic loss variable
                critic_loss_on_model = model.critic(interpolated_last_hidden_state).mean() * self.args.advisery_weight / interpolated_last_hidden_state.size(0)
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
            critic_loss /= 2 * latent.size(0)
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

        if (hasattr(model, 'critic') and model.critic) or self.args.cycle_loss:
            pos = self.args.train_batch_size * (self.state.global_step % self.args.interpolate_training_step_rate)
            self.latent_stack[pos:pos + self.args.train_batch_size] = outputs.latent.detach()
            self.final_decoder_hidden_state_stack[pos:pos + self.args.train_batch_size] = outputs.decoder_hidden_states[-1].detach()
            if self.state.global_step > 0 and pos == 0:
                self.training_interpolation_step(self.final_decoder_hidden_state_stack, self.latent_stack, model)

        return self.get_loss_grad(outputs, labels)

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
