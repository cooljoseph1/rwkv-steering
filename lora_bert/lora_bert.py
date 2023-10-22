import torch.nn as nn

class LoraBert(nn.Module):
    def __init__(self, bert_layers, up_size=2560, down_size=768, k=50, residual_multiplier=1.0):
        super().__init__()
        self.lora_downs = nn.ModuleList([
            Lora(in_size=up_size, out_size=down_size, k=k)
            for _ in bert_layers
        ])

        self.chopped_bert_layers = nn.ModuleList([
            ChoppedBertLayer(bert_layer, up_size=up_size)
            for bert_layer in bert_layers
        ])
        self.residual_multiplier = residual_multiplier

    def forward(self, embeddings):
        up = embeddings
        down = self.lora_downs[0](up)
        for i in range(len(self.chopped_bert_layers) - 1):
            up_residual = self.chopped_bert_layers[i](down)
            up = up + up_residual * self.residual_multiplier
            down_residual = self.lora_downs[i + 1](up)
            down = down + down_residual * self.residual_multiplier

        up_residual = self.chopped_bert_layers[-1](down)
        up = up + up_residual * self.residual_multiplier
        return up


class Lora(nn.Module):
    def __init__(self, in_size=2560, out_size=768, k=50):
        super().__init__()
        self.in_to_k = nn.Linear(in_size, k)
        self.k_to_out = nn.Linear(k, out_size)

    def forward(self, embeddings):
        x = self.in_to_k(embeddings)
        x = self.k_to_out(x)
        return x
    

class ChoppedBertLayer(nn.Module):
    def __init__(self, bert_layer, up_size=2560):
        super().__init__()
        self.attention = bert_layer.attention # frozen
        self.intermediate = bert_layer.intermediate # frozen
        self.output = BertLayerOutput(out_size=up_size) # Not frozen

    def forward(self, embeddings):
        x = self.attention(embeddings)[0]
        x = self.intermediate(x)
        x = self.output(x)
        return x

class BertLayerOutput(nn.Module):
    def __init__(self, in_size=3072, out_size=2560):
        super().__init__()
        self.layer_norm = nn.LayerNorm((in_size,))
        self.dense = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, embeddings):
        x = self.layer_norm(embeddings)
        x = self.dense(x)
        x = self.dropout(x)
        return x