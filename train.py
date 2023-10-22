from tqdm import tqdm
from datasets import load_dataset
import rwkv_model
from rwkv_model import RwkvBot, encode, decode, forward
import torch
from torch.utils.data import DataLoader

device = "cpu"
num_epochs = 10
batches_per_epoch = 10_000

## Prepare Data ##

seed, buffer_size = 1, 1_000
dataset = load_dataset('c4', 'en', split='train', streaming=True)
dataset = dataset.shuffle(seed, buffer_size=buffer_size)
dataloader = DataLoader(dataset, batch_size=64)

def embed(batch):
    """
    Converts a batch to RWKV space.

    Batch looks like:
        { 'text': [
            "passage 0",
            "passage 1",
                ...
        ]}
    
    It converts the text to tokens, truncating extras so they have the same length. Then it runs each list of tokens
    through the RWKV model.
    """

    tokens = [rwkv_model.encode(text) for text in batch['text']]

    # Truncate excess tokens
    min_length = min(map(len, tokens))
    tokens = torch.stack([t[:min_length] for t in tokens])

    with torch.no_grad():
        return rwkv_model.forward(tokens, None)

## Training ##
from augmented_bert import bert_model
import torch.nn.functional as F

bert_model = bert_model.train().to(device)
optim = torch.optim.AdamW(params=bert_model.parameters(), lr=1e-5)
for epoch in range(10):
    dataset.set_epoch(epoch)
    pbar = tqdm(dataloader, total=batches_per_epoch)
    for i, batch in enumerate(pbar):
        embedding = embed(batch)
        noise = (1 + epoch) / num_epochs * (0.5 - torch.rand_like(embedding)) * embedding
        output = bert_model(embedding + noise)

        loss = F.mse_loss(output, noise)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % 10 == 0:
            print(f"loss: {loss.item()}")

        if i % 100 == 0: # Save the model
            pass
