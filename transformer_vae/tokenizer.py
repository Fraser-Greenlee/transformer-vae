from tokenizers import Tokenizer


class MatchPretrainedTokenizer():
    '''
        Subclass to have it work with PretrainedTokenizer's from the transformers library.
    '''
    def __init__(self, max_seq_size, tokenizer_name) -> None:
        self.tokenizer = Tokenizer.from_file(f'tokenizers/{tokenizer_name}')

    @property
    def mask_token(self):
        if self.tokenizer.padding:
            return self.tokenizer.padding['length']

    @mask_token.setter
    def mask_token(self, val):
        self.tokenizer.enable_padding(pad_token='<pad>', pad_id=self.tokenizer.token_to_id('<pad>'), length=val)

    @property
    def model_max_length(self):
        if self.tokenizer.padding:
            return self.tokenizer.padding['length']

    @model_max_length.setter
    def model_max_length(self, val):
        self.tokenizer.enable_padding(pad_token='<pad>', pad_id=self.tokenizer.token_to_id('<pad>'), length=val)

    def __len__(self):
        return self.tokenizer.get_vocab_size()
