from rwkv_model import RwkvBot, encode, decode, forward
from augmented_bert import bert_model
import torch

def get_state_batches(texts):
    """
    A generator to yield batches of RWKV states from a list of texts. The texts are truncated to the one with the shortest
    number of tokens.
    """

    tokens = [encode(text) for text in texts]
    states = []
    state = None
    for token_batch in zip(*tokens):
        token_batch = torch.stack(token_batch).unsqueeze(1)
        print(token_batch)
        _, state = forward(token_batch, state)
        device, dtype = state[0].device, state[0].dtype
        stacked_state = torch.stack(state).to(device=device, dtype=dtype)
        print(stacked_state.shape)
        yield stacked_state

texts = [
    "Why hello world!",
    "This is a second text.",
    "First comment. ~anon",
]

for batch in get_state_batches(texts[:]):
    print(batch)
    with torch.no_grad():
        print(bert_model(batch))