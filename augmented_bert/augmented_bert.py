import torch.nn as nn
from torch import save, load

class AugmentedBert(nn.Module):
    def __init__(self, bert_layers, up_size=2560, down_size=1024, k=50, residual_multiplier=1.0):
        super().__init__()
        self.downs = nn.ModuleList([
            Down(in_size=up_size, out_size=down_size)
            for _ in bert_layers
        ])

        self.chopped_bert_layers = nn.ModuleList([
            ChoppedBertLayer(bert_layer, up_size=up_size)
            for bert_layer in bert_layers
        ])
        self.residual_multiplier = residual_multiplier

    def save_fc(self, path):
        model_dict = {
            'downs': self.downs,
            'ups': [l.output for l in self.chopped_bert_layers]
        }

        save(model_dict, path)

    def load_fc(self, path):
        model_dict = load(path)

        del self.downs # for garbage collection
        self.downs = model_dict['downs']

        for i, l in enumerate(self.chopped_bert_layers):
            del l.output
            l.output = model_dict['ups'][i]

    def forward(self, embeddings):
        up = embeddings
        down = self.downs[0](up)
        for i in range(len(self.chopped_bert_layers) - 1):
            up_residual = self.chopped_bert_layers[i](down)
            up = up + up_residual * self.residual_multiplier
            down_residual = self.downs[i + 1](up)
            down = down + down_residual * self.residual_multiplier

        up_residual = self.chopped_bert_layers[-1](down)
        up = up + up_residual * self.residual_multiplier
        return up
    
class Down(nn.Module):
    def __init__(self, in_size=2560, out_size=1024):
        super().__init__()
        self.layer_norm = nn.LayerNorm((in_size,))
        self.dense = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, embeddings):
        x = self.layer_norm(embeddings)
        x = self.dense(x)
        x = self.dropout(x)
        return x
    

class ChoppedBertLayer(nn.Module):
    def __init__(self, bert_layer, down_size=1024, up_size=2560):
        super().__init__()
        # Note: you should be able to find down_size with
        # bert_layer.attention._modules['self'].query.in_features

        self.attention = bert_layer.attention # frozen
        self.intermediate = bert_layer.intermediate # frozen
        self.output = BertLayerOutput(out_size=up_size) # Not frozen

    def forward(self, embeddings):
        x = self.attention(embeddings)[0]
        x = self.intermediate(x)
        x = self.output(x)
        return x

class BertLayerOutput(nn.Module):
    def __init__(self, in_size=4096, out_size=2560):
        super().__init__()
        self.layer_norm = nn.LayerNorm((in_size,))
        self.dense = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, embeddings):
        x = self.layer_norm(embeddings)
        x = self.dense(x)
        x = self.dropout(x)
        return x