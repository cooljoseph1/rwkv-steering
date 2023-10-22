from .rwkv import RWKV_RNN

class Args:
    pass

args = Args()
args.MODEL_NAME = "./rwkv_model/rwkv-3b/RWKV-4-Pile-3B-Instruct-test1-20230124"
args.RUN_DEVICE = "cuda"
args.ctx_len = 1024

rwkv_rnn = RWKV_RNN(args)

test_tokens = [[10, 20, 6], [3, 40, 50]]
output = rwkv_rnn.forward(test_tokens, None)
print(output.shape)
test_tokens = [[10, 20, 6], [3, 40, 50], [4, 3, 1], [2, 3, 10]]
output = rwkv_rnn.forward(test_tokens, None)
print(output.shape)