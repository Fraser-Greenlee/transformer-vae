import os
import logging
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict
from tqdm.auto import tqdm, trange
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
from unittest.mock import MagicMock

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, TrainOutput


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class T5_VAE_Trainer(Trainer):
    """
    Class for training T5-VAE.
    """

    tokenizer = None
    start_training_mode_step = 0
    log_stores = {
        "decoder_ce": [],
        "decoder_ce_sum": [],
        "recon_loss": [],
        "reg_loss": [],
        "reg_loss_w": [],
    }

    def _setup_wandb(self):
        # Overriding this to get all training args in the run.
        pass

    def _regulariser_loss_weight_schedule(self):
        if self.args.reg_constant_weight is not None:
            return self.args.reg_constant_weight
        return torch.sigmoid(
            torch.tensor(self.global_step * self.args.reg_schedule_k - self.args.reg_schedule_b)
        ).item()

    def _run_training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], log=True) -> float:
        input_ids = inputs["input_ids"].to(self.args.device)

        decoder_ce, recon_loss, reg_loss = model(input_ids)

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce.sum() + reg_loss * reg_loss_w
        if self.args.use_recon_loss:
            loss += recon_loss

        if log and self.is_world_master():
            self.log_stores["decoder_ce_sum"].append(decoder_ce.sum().detach() / input_ids.size(0))
            self.log_stores["decoder_ce"].append(decoder_ce.mean().detach())
            self.log_stores["recon_loss"].append(recon_loss.detach())
            self.log_stores["reg_loss"].append(reg_loss.detach())
            self.log_stores["reg_loss_w"].append(reg_loss_w)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def _training_step(self, *args) -> float:
        loss = self._run_training_step(*args)
        torch.cuda.empty_cache()
        return loss

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log all loss components seperately.
        Seperated to remove use of TB-Writer
        """
        for k, v in self.log_stores.items():
            if len(v):
                logs[k] = sum(v) / float(len(v))
            else:
                logs[k] = 0
            self.log_stores[k] = []

        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)

    def save_model(self, output_dir: Optional[str] = None):
        if self.is_world_master():  # Always save the tokenizer with the model
            self.model.tokenizer.save_pretrained(output_dir if output_dir is not None else self.args.output_dir)
        super().save_model(output_dir)

    def _get_epoch_iterator(self, train_dataloader):
        return tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset)
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size * self.args.n_gpu,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )

        return data_loader

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.
        Needed to add to fix the len(epoch_iterator) adding:
        ```
            epoch_iterator.__len__ = lambda: len(train_dataloader)
        ```

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # multi-gpu training
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = (
                    self.global_step * self.args.gradient_accumulation_steps % len(train_dataloader)
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master())
        for epoch in train_iterator:
            model.train()
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = self._get_epoch_iterator(train_dataloader)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    epoch_iterator.__len__() <= self.args.gradient_accumulation_steps
                    and (step + 1) == epoch_iterator.__len__()
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = self.global_step / (epoch_iterator.__len__() / self.args.gradient_accumulation_steps)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss = tr_loss.item()
                        logs["loss"] = (tr_loss - logging_loss) / (
                            self.args.logging_steps * self.args.gradient_accumulation_steps
                        )
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = scheduler.get_last_lr()[0]
                        logging_loss = tr_loss

                        if self.is_world_master():
                            self._log(logs)

                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self.model.tokenizer.save_pretrained(self.args.output_dir)
                            self._rotate_checkpoints()

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)


@dataclass
class MyTrainingArguments(TrainingArguments):
    project_name: str = field(default=None, metadata={"help": "The Weights & Biases project name for the run."})
    reg_schedule_k: float = field(
        default=0.0025,
        metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."},
    )
    reg_schedule_b: float = field(
        default=6.25,
        metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."},
    )
    reg_constant_weight: Optional[float] = field(
        default=None, metadata={"help": "Apply a constant weight to the regulariser."}
    )
    use_recon_loss: bool = field(
        default=False, metadata={"help": "Have the reconstructed encodings match their input encodings."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    t5_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the T5 model being using for encoding & decoding."},
    )
    model_type: Optional[str] = field(
        default="t5",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    ae_latent_size: int = field(
        default=None, metadata={"help": "The size of the VAE's latent space, only valid with a T5 model."}
    )
    set_seq_size: int = field(default=None, metadata={"help": "Set sequence size, needed for VAE compression."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, set_seq_size, local_rank=-1):
    file_path = args.train_data_file
    return SetSizeLineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        set_seq_size=set_seq_size,
        overwrite_cache=args.overwrite_cache,
        local_rank=local_rank,
    )


def _log_load_failures(model, missing_keys, unexpected_keys, error_msgs):
    if len(missing_keys) > 0:
        logger.info(
            "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
        )
    if len(unexpected_keys) > 0:
        logger.info(
            "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
        )
    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
        )


def _get_ae_encoder_decoder(t5_model_config, model_args, training_args):
    args = (t5_model_config.d_model, model_args.set_seq_size, model_args.ae_latent_size, training_args)
    return LatentEncoderLargeTanh_1kLatent(*args), LatentDecoderLargeT5NormFF(*(args + (t5_model_config,)))


def _get_config(model_args):
    if model_args.config_name:
        return AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_path:
        return AutoConfig.from_pretrained(model_args.model_path, cache_dir=model_args.cache_dir)
    else:
        logger.warning("You are instantiating a new config instance from scratch.")
        return CONFIG_MAPPING[model_args.model_type]()


def _get_t5_model(t5_model_name, tokenizer_name=None, cache_dir=None):
    # Load pretrained model and tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    elif t5_model_name:
        tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir=cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config = AutoConfig.from_pretrained(t5_model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def _get_ae(t5_model_config, model_args, training_args):
    encoder, decoder = _get_ae_encoder_decoder(t5_model_config, model_args, training_args)
    return SeqVAE(encoder, decoder, training_args)


def _get_t5_vae_requirements(model_args, training_args):
    config = _get_config(model_args)
    t5_model, tokenizer = _get_t5_model(model_args.t5_model_name, model_args.tokenizer_name, model_args.cache_dir)
    vae = _get_ae(t5_model.config, model_args, training_args)
    return config, t5_model, tokenizer, vae


def new_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    return t5_VAE(config, t5_model, vae, model_args.set_seq_size, tokenizer)


def load_t5_vae(model_args, training_args):
    config, t5_model, tokenizer, vae = _get_t5_vae_requirements(model_args, training_args)
    return t5_VAE.from_pretrained(
        model_args.model_path,
        config=config,
        t5_model=t5_model,
        vae=vae,
        set_seq_size=model_args.set_seq_size,
        tokenizer=tokenizer,
        cache_dir=model_args.cache_dir,
    )


def load_t5_vae_from_args(args_list):
    # Use to load a T5_VAE from a jupyter notebook
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, _, training_args = parser.parse_args_into_dataclasses(args=args_list)
    assert model_args.model_path and os.path.isdir(model_args.model_path)
    return load_t5_vae(model_args, training_args)


def main(alt_local_rank=None):

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if alt_local_rank is not None:
        training_args.local_rank = alt_local_rank

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    if model_args.model_path and os.path.isdir(model_args.model_path):
        model = load_t5_vae(model_args, training_args)
    else:
        model = new_t5_vae(model_args, training_args)

    # Get datasets
    train_dataset = (
        get_dataset(
            data_args,
            tokenizer=model.tokenizer,
            set_seq_size=model_args.set_seq_size,
            local_rank=training_args.local_rank,
        )
        if training_args.do_train
        else None
    )
    data_collator = Seq2SeqDataCollatorForLanguageModeling(tokenizer=model.tokenizer)
    data_collator.mlm = False

    trainer = T5_VAE_Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        prediction_loss_only=True,
        tb_writer=MagicMock(),
    )

    if trainer.is_world_master():
        wandb.init(
            project=training_args.project_name,
            name=training_args.output_dir,
            config={**vars(training_args), **vars(data_args), **vars(model_args)},
        )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_path
            if model_args.model_path is not None and os.path.isdir(model_args.model_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            model.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
