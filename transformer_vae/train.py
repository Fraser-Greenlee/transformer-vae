"""
    Train Transformer-VAEs using the Huggingface Trainer with Weights and Biasis.
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import is_wandb_available
from transformers.trainer_utils import is_main_process

from transformer_vae.trainer import VAE_Trainer
from transformer_vae.data_collator import DataCollatorForLanguageAutoencoding
from transformer_vae.trainer_callback import TellModelGlobalStep
from transformer_vae.model import Funnel_T5_VAE_Model
from transformer_vae.sequence_checks import SEQ_CHECKS
from transformer_vae.config import Funnel_T5_VAE_Config


logger = logging.getLogger(__name__)


@dataclass
class VAE_TrainingArguments(TrainingArguments):
    """
    Extra arguments to specify generation during evaluation.
    """

    generate_min_len: int = field(
        default=1,
        metadata={"help": "The minimum length of sequences to be generated from latent points during evaluation."},
    )
    generate_max_len: int = field(
        default=20,
        metadata={"help": "The maximum length of sequences to be generated from latent points during evaluation."},
    )
    seq_check: str = field(
        default=None,
        metadata={"help": f"Run check on sequences from random latent codes. Options: {', '.join([str(k) for k in SEQ_CHECKS.keys()])}"},
    )
    max_validation_size: int = field(
        default=None,
        metadata={"help": "Limit the eval dataset size, defaults to not limiting it, must be < validation size."},
    )
    sample_from_latent: bool = field(
        default=False,
        metadata={"help": "Whether to sample from the latent space during evaluation."},
    )
    test_classification: bool = field(
        default=False,
        metadata={"help": "Test using latent codes for unsupervised classification."},
    )
    cycle_loss: bool = field(
        default=False,
        metadata={"help": "Encourage the encoder & decoder to produce a bijective mapping. Feeds the final decoder hidden state to the encoder and compares the latent codes."},
    )
    vae_cycle_loss: bool = field(
        default=False,
        metadata={"help": "Encourage the encoder & decoder to produce a bijective mapping. Feeds the final decoder hidden state to the encoder and compares the latent codes."},
    )
    advisery_weight: int = field(
        default=1,
        metadata={"help": "Encourage the encoder & decoder to produce a bijective mapping. Feeds the final decoder hidden state to the encoder and compares the latent codes."},
    )
    cycle_weight: int = field(
        default=1,
        metadata={"help": "Encourage the encoder & decoder to produce a bijective mapping. Feeds the final decoder hidden state to the encoder and compares the latent codes."},
    )
    interpolate_training_step_rate: int = field(
        default=1,
        metadata={"help": "Run a batch of iterpolation losses every N steps."},
    )
    min_critic_steps: int = field(
        default=1_000,
        metadata={"help": "Start updating the model with the critic loss after N steps."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    set_seq_size: int = field(metadata={"help": "Set sequence size, must be set."})
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    transformer_name: Optional[str] = field(
        default='funnel-transformer/intermediate',
        metadata={
            "help": "Name of the transformer model being using for encoding & decoding (must be a variant on the funnel transformer)."
        },
    )
    transformer_decoder_name: Optional[str] = field(
        default="t5-base",
        metadata={
            "help": "Name of the transformer model being using for decoding the funnel transformer (only `t5` type transformers supported)."
        },
    )
    transformer_critic_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the transformer model being using to critique interpolated hidden states, works with T5 & Funnel transformers."
        },
    )
    critic_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Kind of critic model to use."
        },
    )
    config_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='t5-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    latent_size: int = field(default=1_000, metadata={"help": "The size of the VAE's latent space."})
    encoded_seq_size: int = field(
        default=None, metadata={"help": "Sequence size of encoded sequence, needed for Funnel-VAE."}
    )
    encoder_model: Optional[str] = field(
        default=None, metadata={"help": "Name of the model that converts hidden states into latent codes."}
    )
    decoder_model: Optional[str] = field(
        default=None, metadata={"help": "Name of the model that converts latent codes into hidden states."}
    )
    # Arguments used during training
    mmd_batch_size: int = field(
        default=None,
        metadata={
            "help": "Run multuple, smaller batches for MMD-VAE regularisation loss (training batch size must be divisible by the MMD batch size)."
        },
    )
    dont_use_reg_loss: bool = field(
        default=False,
        metadata={
            "help": "Toggle regularisation loss, without regularisation your training an autoencoder rather than a VAE."
        },
    )
    reg_schedule_k: float = field(
        default=0.0025,
        metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."},
    )
    reg_schedule_b: float = field(
        default=6.25,
        metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."},
    )
    n_latent_tokens: int = field(
        default=5,
        metadata={
            "help": "Number of tokens to use when making a sequence latent code."
        },
    )
    tye_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Have the Encoder embedding use the same weights as the decoder embedding."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(default=None, metadata={"help": "Use this dataset column as 'text'."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.0, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    validation_name: str = field(
        default="validation",
        metadata={"help": "Name of the set to run evaluation on."},
    )
    classification_column: str = field(
        default="class_label",
        metadata={"help": "Test SVM classification using latent codes."},
    )
    num_classes: int = field(
        default=None,
        metadata={"help": "How many classes in the data, found using a ClassLabel column if none given."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VAE_TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # use train batch size if eval is default
    training_args.per_device_eval_batch_size = (
        training_args.per_device_train_batch_size
        if training_args.per_device_eval_batch_size == 8
        else training_args.per_device_eval_batch_size
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    return model_args, data_args, training_args


def setup_logs(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def get_datasets(data_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        return load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files[data_args.validation_name] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    # TODO have this dataset split samples on some substring
    return load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


def load_model_and_tokenizer(model_args):
    # Distributed training:
    # The `.from_pretrained` methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.set_seq_size and model_args.set_seq_size <= 4:
        logger.warn('`set_seq_size` is to small to work with the Funnel transformer. now using set_seq_size=5')
        model_args.set_seq_size = 5

    if model_args.config_path:
        config = Funnel_T5_VAE_Config.from_pretrained(
            model_args.config_path, cache_dir=model_args.cache_dir
        )
    elif model_args.model_path:
        config = Funnel_T5_VAE_Config.from_pretrained(
            model_args.model_path, cache_dir=model_args.cache_dir
        )
    else:
        config = Funnel_T5_VAE_Config(
            latent_size=model_args.latent_size,
            transformer_name=model_args.transformer_name,
            transformer_decoder_name=model_args.transformer_decoder_name,
            transformer_critic_name=model_args.transformer_critic_name,
            critic_type=model_args.critic_type,
            set_seq_size=model_args.set_seq_size,
            encoded_seq_size=model_args.encoded_seq_size,
            mmd_batch_size=model_args.mmd_batch_size,
            use_reg_loss=(not model_args.dont_use_reg_loss),
            reg_schedule_k=model_args.reg_schedule_k,
            reg_schedule_b=model_args.reg_schedule_b,
            n_latent_tokens=model_args.n_latent_tokens,
            tye_embeddings=model_args.tye_embeddings,
            encoder_model=model_args.encoder_model,
            decoder_model=model_args.decoder_model,
            use_extra_logs=is_wandb_available(),
        )
        logger.warning("You are instantiating a new config instance from scratch (still using T5 checkpoint).")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.transformer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.transformer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_path:
        model = Funnel_T5_VAE_Model.from_pretrained(
            model_args.model_path,
            from_tf=bool(".ckpt" in model_args.model_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = Funnel_T5_VAE_Model(config)

    model.resize_token_embeddings(len(tokenizer))
    if model_args.set_seq_size:
        tokenizer.model_max_length = model_args.set_seq_size
    tokenizer.mask_token = tokenizer.unk_token

    return model, tokenizer


def preprocess_datasets(training_args, data_args, tokenizer, datasets):
    # Add class_label if needed
    if training_args.test_classification:
        if data_args.classification_column != "class_label":

            if not data_args.num_classes:
                data_args.num_classes = (
                    datasets[data_args.validation_name].features[data_args.classification_column].num_classes
                )

            def add_class_column(examples):
                return {"class_label": examples[data_args.classification_column]}

            datasets = datasets.map(add_class_column, remove_columns=[data_args.classification_column])

    # tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets[data_args.validation_name].column_names
    if data_args.text_column is not None:
        text_column_name = data_args.text_column
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]

    if text_column_name != "text":
        logger.info(f'Using column "{text_column_name}" as text column.')

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding="max_length", truncation=True)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.max_validation_size:
        tokenized_datasets[data_args.validation_name] = tokenized_datasets[data_args.validation_name].train_test_split(
            training_args.max_validation_size
        )["test"]

    data_collator = DataCollatorForLanguageAutoencoding(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    return data_collator, tokenized_datasets


def main():
    model_args, data_args, training_args = get_args()

    setup_logs(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = get_datasets(data_args)

    model, tokenizer = load_model_and_tokenizer(model_args)

    data_collator, tokenized_datasets = preprocess_datasets(training_args, data_args, tokenizer, datasets)

    # Initialize our Trainer
    trainer = VAE_Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets[data_args.validation_name] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TellModelGlobalStep],
    )
    trainer.log({})

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_path if model_args.model_path and os.path.isdir(model_args.model_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_T5_VAE.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
