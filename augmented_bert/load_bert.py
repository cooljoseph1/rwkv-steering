import os
from transformers import BertModel

# my_dir = os.path.dirname(__file__)
# bert_path = os.path.join(my_dir, "bert-340m")
bert = BertModel.from_pretrained("bert-large-cased")

bert_layers = bert.encoder.layer