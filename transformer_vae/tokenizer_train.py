'''
    Train a BPE on a given Huggingface dataset.
'''
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer
from transformers import HfArgumentParser


TOKENIZERS = {
    'byte': ByteLevelBPETokenizer,
    'char': CharBPETokenizer
}
TMP_FILE = 'secret_tmp_dataset_file_delete_me.txt'


@dataclass
class TokenizerArguments:
    dataset: str = field(
        metadata={"help": "Dataset name."}
    )
    name: str = field(
        metadata={"help": "Name of your new tokenizer, defaults to dataset + type."},
        default=''
    )
    dataset_col: str = field(
        default='text',
        metadata={"help": "Column to train on."}
    )
    dataset_seg: str = field(
        default='train',
        metadata={"help": "Dataset segment to train on."}
    )
    type: str = field(
        default='byte',
        metadata={"help": f"Type of tokenizer to train. ({TOKENIZERS.keys()})"}
    )
    special_tokens: str = field(
        default='</s> <pad>',
        metadata={"help": "Tokenizer special tokens (seperate with spaces)."}
    )
    vocab_size: int = field(
        default=32128,
        metadata={"help": "Max vocab size."}
    )
    min_frequency: int = field(
        default=2,
        metadata={"help": "Min token frequency."}
    )


def main():
    args = HfArgumentParser(TokenizerArguments).parse_args_into_dataclasses()[0]
    dataset = load_dataset(args.dataset)

    txt = []
    for r in tqdm(dataset[args.dataset_seg], desc='writing dataset to temp file'):
        txt.append(r[args.dataset_col])

    with open(TMP_FILE, 'w') as f:
        f.write('\n'.join(txt))

    tokenizer = TOKENIZERS[args.type]()
    tokenizer.train(
        TMP_FILE,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens.split()
    )
    os.remove(TMP_FILE)

    name = args.name if args.name else f'tkn_{args.dataset.split("/")[::-1][0]}_{args.type}'
    tokenizer.save(f'tokenizers/{name}.json')

    print('### Show performance')
    for i, r in enumerate(dataset[args.dataset_seg]):
        txt = r[args.dataset_col]
        token_ids = tokenizer.encode(txt).ids
        if i == 0:
            print('real:')
            print(txt)
            print('tokenizer decode( encode( txt ) ):')
            print(tokenizer.decode(token_ids))
            print('list tokens:')
            print('", "'.join([tokenizer.decode([tkn_id]) for tkn_id in token_ids]))
        if i < 10:
            print('raw len: ', len(txt))
            print('split len:', len(txt.split()))
            print('tokenized len:', len(token_ids))
        else:
            break


if __name__ == '__main__':
    main()
