import logging
from typing import Dict, List
import random
import time
import pickle
import os
import torch
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)


logger = logging.getLogger(__name__)


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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
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


















class SetSizeLineByLineTextDataset(Dataset):
    """
    Same as `LineByLineTextDataset` by Huggingface but modified to used fixed length sequences & to cache the result.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, set_seq_size, overwrite_cache=False, local_rank=-1
    ):
        logger.info("Loading text.")

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_set_size_line_by_line_{tokenizer.__class__.__name__}_set_seq_size_{set_seq_size}_{filename}",
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            logger.info(f"Loading features from cached file {cached_features_file}...")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("[took %.3f s]", time.time() - start)

        else:
            if not os.path.isfile(file_path):
                raise Exception(
                    f"Can't find true file:\n{file_path}\nAlso can't find cahced file:\n{cached_features_file}"
                )
            # don't create cache when running in parallel

            logger.info(f"Creating features from dataset file at {directory}")

            seq_texts = self._get_text_sequences(file_path)
            random.shuffle(seq_texts)

            tokenized_seqs = []
            for text in tqdm(seq_texts, desc="Tokenizing each sequence."):
                tokenized_seqs.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

            logger.info(f"Max sequence length in dataset: {max([len(seq)+1 for seq in tokenized_seqs])}")

            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            skip_count = 0
            self.examples = []
            for tokens in tqdm(tokenized_seqs, desc="Making sequence labels."):
                tokens.append(tokenizer.eos_token_id)
                if len(tokens) > set_seq_size:
                    skip_count += 1
                    continue
                input_tokens = self._pad_tokens(set_seq_size, torch.tensor(tokens), pad_token)
                self.examples.append(tokenizer.build_inputs_with_special_tokens(input_tokens))

            logger.info(f"Got {len(self.examples)} examples.")
            logger.info(f"Skipped {skip_count} examples since they were too long.")

            start = time.time()
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start)

    @staticmethod
    def _get_text_sequences(file_path):
        with open(file_path, encoding="utf-8") as f:
            seq_texts = f.read().split("\n")

        # remove empty strings & strip
        seq_texts = list(filter(None, seq_texts))
        seq_texts = [txt.strip() for txt in seq_texts]

        return seq_texts

    @staticmethod
    def _pad_tokens(set_size, tokens_tensor, pad_token):
        padedd = torch.ones(set_size, dtype=torch.long) * pad_token
        padedd[: tokens_tensor.size(0)] = tokens_tensor
        return padedd

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class Seq2SeqDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    mlm: bool = False

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {"input_ids": input_ids}
