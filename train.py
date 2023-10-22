from tqdm import tqdm
from datasets import load_dataset
import rwkv_model
from rwkv_model import RwkvBot, encode, decode, forward
import torch
from torch.utils.data import DataLoader

device = "cuda"
num_epochs = 10
batches_per_epoch = 10_000
save_path = "denoiser.fc"

## Prepare Data ##

seed, buffer_size = 1, 1_000
dataset = load_dataset('c4', 'en', split='train', streaming=True)
dataset = dataset.shuffle(seed, buffer_size=buffer_size)
dataloader = DataLoader(dataset, batch_size=8)

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
    tokens = [t[:min_length] for t in tokens]

    embeddings = None
    for token_batch in zip(*tokens):
        token_batch = torch.tensor(token_batch).unsqueeze(0)
        with torch.no_grad():
            embeddings = rwkv_model.forward(token_batch, embeddings)
        yield embeddings

## Training ##
from augmented_bert import bert_model
import torch.nn.functional as F
import os

if os.path.exists(save_path):
    bert_model.load_fc(save_path)

bert_model = bert_model.train().to(device, dtype=torch.bfloat16)

optim = torch.optim.AdamW(params=bert_model.get_trainable_parameters(), lr=1e-4)
for epoch in range(10):
    dataset.set_epoch(epoch)
    pbar = tqdm(dataloader, total=batches_per_epoch)
    for i, batch in enumerate(pbar):
        loss = torch.tensor(0).to(device='cuda', dtype=torch.bfloat16)
        for embedding in embed(batch):
            embedding = embedding.to(dtype=torch.bfloat16)
            noise = (1 + epoch) / num_epochs * (0.5 - torch.rand_like(embedding))
            noisy = embedding + noise * embedding
            noisy = (noisy - noisy.mean()) / noisy.std()

            output = bert_model(noisy)

            loss = F.mse_loss(output, noise)
            loss.backward()
            optim.step()
            optim.zero_grad()

            print(f"loss: {loss.item()}")

        if i % 10 == 0: # Save the model
            bert_model.save_fc(save_path)
