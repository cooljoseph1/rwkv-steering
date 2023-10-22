from .lora_bert import LoraBert
from .load_bert import bert_layers as _bert_layers
_bert_layers.requires_grad_(False) # Freeze the BERT model

lora_bert_model = LoraBert(_bert_layers)