import time
from tqdm import tqdm
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
import torch
from torch import nn, autograd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers import trainer as trainer_script
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
    hp_params
)
from transformers.file_utils import WEIGHTS_NAME, is_apex_available, speed_metrics
if is_apex_available():
    from apex import amp

from transformer_vae.sequence_checks import SEQ_CHECKS
from transformer_vae.trainer_callback import WandbCallbackUseModelLogs
from transformer_vae.sklearn import train_svm
from transformer_vae.utils import slerp


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
    def __init__(self, **kwargs):
        self.critic = kwargs.critic
        super().__init__(**kwargs)

    def _text_from_latent(self, latent):
        # TODO can I do many latents in a batch?
        generation = self.model.generate(
            latent=latent, bos_token_id=self.model.decoder_start_token_id, min_length=self.args.generate_min_len, max_length=self.args.generate_max_len
        )
        return self.tokenizer.decode(generation[0].tolist(), skip_special_tokens=True)

    def _interpolate_samples(self, eval_dataset):
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
        start_latent, end_latent = latents[0].view(1, -1), latents[1].view(1, -1)

        seq_check_results = 0
        seq_check = SEQ_CHECKS[self.args.seq_check]
        table = wandb.Table(columns=["Interpolation Ratio", "Text", "Valid"])
        table.add_data(-10, self.tokenizer.decode(samples["input_ids"][0]), True)

        for i in tqdm(range(11), desc="Sampling from interpolated latent points"):
            ratio = i / 10
            latent = slerp(ratio, start_latent, end_latent)
            text = self._text_from_latent(latent)
            valid = seq_check(text)
            table.add_data(ratio, text, valid)
            if ratio > 0 and i < 1:
                seq_check_results += int(valid)

        table.add_data(10, self.tokenizer.decode(samples["input_ids"][1]), True)
        wandb.log({"interpolate points": table}, step=self.state.global_step)
        if self.args.seq_check:
            wandb.log(
                {'interpolation samples passing seq check': seq_check_results / 9},
                step=self.state.global_step
            )

    def _random_samples(self):
        # TODO can I greedy decode these in parallel?
        table = wandb.Table(columns=["Text", "Valid"])
        latent_points = torch.randn(self.args.n_random_samples, self.model.config.latent_size, device=self.model.device)
        seq_check_results = 0
        seq_check = SEQ_CHECKS[self.args.seq_check]

        for i in tqdm(range(latent_points.size(0)), desc="Sampling from random latent points"):
            text = self._text_from_latent(latent_points[i].view(1, -1))
            valid = seq_check(text)
            table.add_data(text, valid)
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
        wandb.log({"my_custom_id" : wandb.plot.scatter(table, "x", "y", "class")})
        """
        pass

    def _evaluate_latent_samples(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.args.sample_from_latent:
            self._interpolate_samples(eval_dataset)
            self._random_samples()
        if self.args.test_classification:
            latents_with_class = self._latent_with_class(eval_dataset)
            self._svm_classification(latents_with_class)
        # self._t_sne(latents_with_class)

    def get_optimizer(self, a_model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in a_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in a_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        if self.sharded_dpp:
            raise NotImplementedError()
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        Subclassed to use another optimizer and LR scheduler for advisery.
        """
        assert self.optimizer is None
        self.optimizer = self.get_optimizer(self.model)
        if self.critic:
            self.adv_optimizer = self.get_optimizer(self.critic)

        assert self.lr_scheduler is None
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        if self.adv_optimizer:
            self.adv_lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.adv_optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

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

    def interpolation_inputs(self, latent):
        batch_size = latent.size(0)
        interpolation_ratios = torch.rand(batch_size) * 0.5
        # TODO allow sphlerp interpolation here
        shifted_indices = torch.arange(latent.size(0))[1:].tolist() + [0]
        latent_interpolated = slerp(interpolation_ratios, latent, latent[shifted_indices])
        return latent_interpolated, interpolation_ratios

    def prepare_interpolation_data(self, outputs, model):
        '''
        For optimising model interpolations directly, find interpolated latent codes with their ratio.
        Produces 1 interpolation for every 2 samples.
        '''
        interp_ratio, interp_latent = self.interpolation_inputs(outputs.latent)
        # TODO greedily decode interp_latent to get interp_decoder_hidden, preferably in parallel
        generation = self.model.generate(latent=interp_latent, bos_token_id=self.model.decoder_start_token_id, min_length=self.args.generate_min_len, max_length=self.args.generate_max_len)
        # TODO return final decoder hidden states using this

    def training_interpolation_step(self, outputs, model):
        # TODO add adviserial interpolation loss here?
        interpolated_last_hidden_state, target_a = self.prepare_interpolation_data(outputs, model)
        interpolated_last_hidden_state_d = interpolated_last_hidden_state.detach()

        # update model
        with torch.no_grad():
            # don't acumulate gradients on critic model
            disc_loss_no_grad = self.critic(interpolated_last_hidden_state_d)
            # calculate gradient manually
            disc_loss_to_last_hidden = autograd.grad(disc_loss_no_grad, interpolated_last_hidden_state_d)
            # acumulate gradient in VAE model (will only be the VAE-decoder)
            interpolated_last_hidden_state.backward(disc_loss_to_last_hidden)

        # update critic
        # real samples
        d_final_decoder_hidden_state = outputs.decoder_hidden_states[:, -1].detach()
        adv_loss = self.critic(d_final_decoder_hidden_state, torch.zeros(d_final_decoder_hidden_state.size(0)))
        # interpolate samples
        adv_loss += self.critic(interpolated_last_hidden_state_d, target_a)
        adv_loss.backward()  # accumulate gradient on advisery

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Adds adviserial loss when using critic model.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        self.training_interpolation_step(outputs, model)

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
        if is_wandb_available():
            start_eval = time.time()
            with torch.no_grad():
                self.model.eval()
                self._evaluate_latent_samples(eval_dataset=eval_dataset)
            generate_time = time.time() - start_eval
        output_metrics = super().evaluate(eval_dataset=eval_dataset)
        if is_wandb_available():
            self.log({"eval_get_test_loss_time": time.time() - start_eval + generate_time})  # type: ignore
            self.log({"eval_generate_time": generate_time})  # type: ignore
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

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.
        Having to subclass to add discriminator optimizer & LR scheduler.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            raise NotImplementedError()
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            raise NotImplementedError()
            '''
            if self.adv_optimizer:
                model, [self.optimizer, self.adv_optimizer] = amp.initialize(model, [self.optimizer, self.adv_optimizer], opt_level=self.args.fp16_opt_level)
            else:
                model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)
            '''

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_dpp:
            raise NotImplementedError()
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)
                            self.scaler.unscale_(self.adv_optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )
                        if hasattr(self.adv_optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.adv_optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.adv_optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        self.deepspeed.step()
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.step(self.adv_optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        self.adv_optimizer.step()

                    self.lr_scheduler.step()
                    self.adv_lr_scheduler.step()
                    model.zero_grad()
                    self.critic.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
