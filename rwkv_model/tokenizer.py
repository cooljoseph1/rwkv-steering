import os
from transformers import PreTrainedTokenizerFast

my_dir = os.path.dirname(__file__)
RwkvTokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(my_dir, "rwkv-3b", "20B_tokenizer.json"))