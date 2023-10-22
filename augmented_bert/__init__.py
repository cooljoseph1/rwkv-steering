from .augmented_bert import AugmentedBert
from .load_bert import bert_layers as _bert_layers
_bert_layers.requires_grad_(False) # Freeze the BERT model

bert_model = AugmentedBert(_bert_layers)