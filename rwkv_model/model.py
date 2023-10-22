import os

# To use CUDA, change the environ and set `strategy = cuda fp16` (or fp32 if you prefer).

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
from rwkv.model import RWKV                         # pip install rwkv

import rwkv.model


my_dir = os.path.dirname(__file__)
RwkvModel = RWKV(
    model=os.path.join(my_dir, "rwkv-3b", "RWKV-4-Pile-3B-Instruct-test1-20230124.pth"),
    strategy='cpu fp32'
)

RwkvModel.device = RwkvModel.w['emb.weight'].device
RwkvModel.dtype = RwkvModel.w['emb.weight'].dtype