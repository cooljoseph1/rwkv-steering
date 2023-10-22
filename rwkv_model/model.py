import os

# To use CUDA, change the environ and set `strategy = cuda fp16` (or fp32 if you prefer).

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
from .rwkv import RWKV_RNN

my_dir = os.path.dirname(__file__)

class Args:
    pass

args = Args()
args.MODEL_NAME = os.path.join(my_dir, "rwkv-1.5b", "RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096")
args.RUN_DEVICE = "cuda"
args.ctx_len = 1024

RwkvModel = RWKV_RNN(args)

RwkvModel.device = RwkvModel.w.emb.weight.device
RwkvModel.dtype = RwkvModel.w.emb.weight.dtype

# test_tokens = [[10, 20, 6], [3, 40, 50]]
# output = rwkv_rnn.forward(test_tokens, None)
# print(output.shape)
# test_tokens = [[10, 20, 6], [3, 40, 50], [4, 3, 1], [2, 3, 10]]
# output = rwkv_rnn.forward(test_tokens, None)
# print(output.shape)


# from .rwkv.model import RWKV                         # pip install rwkv

# import rwkv.model


# my_dir = os.path.dirname(__file__)
# RwkvModel = RWKV(
#     model=os.path.join(my_dir, "rwkv-3b", "RWKV-4-Pile-3B-Instruct-test1-20230124.pth"),
#     strategy='cpu fp32'
# )

# RwkvModel.device = RwkvModel.w['emb.weight'].device
# RwkvModel.dtype = RwkvModel.w['emb.weight'].dtype