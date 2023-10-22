import torch

from .model import RwkvModel
from .tokenizer import RwkvTokenizer


def encode(text):
    """
    Given a string or list of strings, first split it up into tokens and then encode it.
    
    Returns a Pytorch array (1D if text is a string, 2D if text is a list of strings)
    """
    tokenizer_object = RwkvTokenizer(text, return_tensors="pt")
    ids = tokenizer_object['input_ids']
    if isinstance(text, str):
        return ids[0]
    return ids

def decode(tokens):
    """
    Given tokens, reconstruct the string (if 1D) or list of strings (if 2D)
    """
    try:
        _ = tokens[0]
    except:
        raise ValueError("Must provide a list of tokens or list of lists of tokens")
    
    # Check if 1d or 2d
    is1d = False
    try:
        _ = iter(tokens[0])
    except:
        is1d = True

    if is1d:
        return RwkvTokenizer.decode(tokens)
    else:
        return RwkvTokenizer.batch_decode(tokens)
    
def forward(tokens, hidden_state=None):
    tokens = torch.tensor(tokens, device=RwkvModel.device, dtype=torch.long)
    state = RwkvModel.forward(tokens, hidden_state)
    return state

def sample_token(logits, temperature=1.0, top_k=10):
    top_k = min(top_k, logits.size(-1)) # safety check

    scaled_logits = logits / temperature # scale by temperature
    print("SC_LOG:", scaled_logits)
    # Apply top-k filtering to restrict choices
    values, indices = torch.topk(scaled_logits, top_k, dim=-1)
    scaled_logits[scaled_logits < values[-1]] = float('-inf')
    probabilities = torch.softmax(scaled_logits, dim=0)
    probabilities = probabilities.clamp(0.0, 1.0)
    sampled_index = torch.multinomial(probabilities, 1)
    
    return sampled_index


class ModelWrapper:
    CTX_LENGTH = 1024
    def __init__(self):
        self.state = None
        self.old_state = None
        self.tokens = []
        self.next_logits = None

    def reset(self):
        self.state = None
        self.next_logits = None

    def read(self, text):
        self.tokens = list(encode(text))
        output_logits, new_state = forward(self.tokens, self.state)
        self.state = new_state
        self.next_logits = output_logits

    def _sample_token(self, **kwargs):
        new_token = sample_token(self.next_logits, **kwargs)
        return new_token

    def _get_next_logits(self):
        print("STATE:", torch.cat(self.state).std())
        next_logits, new_state = forward(self.tokens[-1], self.state)
        print("STATE2:", torch.cat(new_state).std())
        return next_logits, new_state

    def write_one(self, temperature=1.0, top_k=10, state_modifier=None):
        new_token = self._sample_token(temperature=temperature, top_k=top_k)
        self.tokens.append(new_token)
        self.tokens = self.tokens[-self.CTX_LENGTH:]
        if state_modifier is not None:
            self.state = state_modifier(self.state)
        self.next_logits, self.state = self._get_next_logits()
        return new_token

    def write(self, num_tokens=1, prompt=None, state_modifier=None, temperature=1.0, top_k=10):
        if prompt is not None:
            self.reset() # Get rid of previous state
            self.read(prompt)

        if self.next_logits is None:
            raise RuntimeError("self.next_logits is not set--you need to tell the model to read something first")
        
        generated_tokens = []
        for i in range(num_tokens):
            new_token = self.write_one(temperature=temperature, top_k=top_k, state_modifier=state_modifier)
            generated_tokens.append(new_token)
            
        generated_text = decode(generated_tokens)
        return generated_text
