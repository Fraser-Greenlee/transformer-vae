from transformers.data.data_collator import (
    torch,
    DataCollatorForLanguageModeling,
    Union,
    PaddingStrategy,
    List,
    Dict,
    Optional,
    Tuple,
    BatchEncoding,
    _collate_batch,
    dataclass,
)


@dataclass
class NonPaddingDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Prevents collator from padding the dataset, needed when using an already tokenized & padded dataset.

    Fix for incorrect default value in `PreTrainedTokenizerBase.pad` arg
    https://github.com/huggingface/transformers/issues/8837
    """

    padding: Union[bool, str, PaddingStrategy] = False

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            # CHANGES START
            batch = self.tokenizer.pad(examples, padding=self.padding, return_attention_mask=False, return_tensors="pt")
            # CHANGES END
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


@dataclass
class DataCollatorForLanguageAutoencoding(NonPaddingDataCollatorForLanguageModeling):
    """
    Same as MLM except we calculate a loss on non-masked tokens.
    """

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # CHANGED line
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
