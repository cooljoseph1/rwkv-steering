from rwkv_model import RwkvBot, encode, decode, forward
import torch
import random

def get_states(text):
    tokens = encode(text)
    states = []
    state = None
    for token in tokens:
        _, state = forward([token], state)
        device, dtype = state[0].device, state[0].dtype
        stacked_state = torch.stack(state).to(device=device, dtype=dtype)
        states.append(stacked_state)
    return states

def get_file_states(file, *args, **kwargs):
    with open(file) as f:
        text = f.read()[:1000]
    return get_states(text, *args, **kwargs)

def get_mean_states(text):
    tokens = encode(text)
    num_tokens = len(tokens)
    mean_state = 0
    state = None
    for token in tokens:
        _, state = forward([token], state)
        device, dtype = state[0].device, state[0].dtype
        stacked_state = torch.stack(state).to(device=device, dtype=dtype)
        mean_state = mean_state + stacked_state / num_tokens
    return mean_state

def get_mean(states):
    return sum([s / len(states) for s in states])

# yoda_states = get_file_states("yoda.txt")
# doi_states = get_file_states("declaration_of_independence.txt")

# mean_yoda = get_mean(yoda_states)
# mean_doi = get_mean(doi_states)

with open("truths.txt", encoding="utf-8") as f:
    mean_truth = get_mean_states(f.read()[:5000])

with open("lies.txt", encoding="utf-8") as f:
    mean_lie = get_mean_states(f.read()[:5000])

# text = open("declaration_of_independence.txt").read()
# capital = get_states(text.upper())
# lower = get_states(text.lower())
# mean_capital = get_mean(capital)
# mean_lower = get_mean(lower)

def anchor_state(state, ground_state, factor=0.2):
    new_state = [s * (1 - factor) + t * factor for s, t in zip(state, ground_state)]
    return new_state

def shift_state(state, diff_state, factor=0.2):
    new_state = [s + t * factor for s, t in zip(state, diff_state)]
    return new_state

def anchor(vector, amount):
    return lambda x: anchor_state(x, vector, amount)

def shift(vector, amount):
    return lambda x: shift_state(x, vector, amount)

def anchor_nn(states, amount, top_k=5):
    batched_states = torch.stack([s.flatten() for s in states])
    batched_states = batched_states / batched_states.square().sum(dim=1, keepdim=True).sqrt()

    def get_closest(state):
        device, dtype = state[0].device, state[0].dtype
        state = torch.cat(state).to(device=device, dtype=dtype)
        dist_cos = (state * batched_states).sum(dim=1)
        _, indices = torch.topk(dist_cos, top_k, dim=-1)
        return [sum(states[i][j] / top_k for i in indices) for j in range(len(states[0]))]
    return lambda x: anchor_state(x, get_closest(x), amount)

def anchor_random(states, amount):
    return lambda x: anchor_state(x, random.choice(states), amount)

#print(''.join(RwkvBot.write(100, "The capital of the United States is", state_modifier=shift(mean_lie - mean_truth, 0.01), temperature=1.0, top_k=5)))