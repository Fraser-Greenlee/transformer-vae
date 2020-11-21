from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel

VOCAB_FILE = "data/tx1_vocab.txt"

with open(VOCAB_FILE, "r") as f:
    words = list(set(f.read().strip().split("\n")))

vocab = {}
for i, word in enumerate(["<pad>", "<unk>"] + words):
    vocab[word] = i

tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
tokenizer.enable_padding(pad_token="<pad>")
tokenizer.pre_tokenizer = Whitespace()

tokenizer.save("data/tokenizer-LakhNES-tx1.json")
