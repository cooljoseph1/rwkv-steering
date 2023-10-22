########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types
import torch
from torch.nn import functional as F
import gc

############################################################################################################



class RWKV_RNN(torch.nn.Module):
    def __init__(self, args):
        """
        Required args:
            args.RUN_DEVICE
            args.MODEL_NAME
        """
        super().__init__()

        self.args = args
        self.FLOAT_MODE = torch.bfloat16
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            args.n_embd = w['emb.weight'].shape[1]
            args.n_layer = 0
            keys = list(w.keys()) # refine weights and send to correct device
            print_need_newline = False
            for x in keys:
                w[x].requires_grad = False
                gc.collect()
                if 'cuda' in args.RUN_DEVICE:
                    torch.cuda.empty_cache()
                if x == 'emb.weight' or 'ln0' in x:
                    continue
                
                block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, block_id+1)
                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
                    w[x] = w[x].t()
                
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].to(dtype=self.FLOAT_MODE)
                
                if 'cuda' in self.RUN_DEVICE:
                    w[x] = w[x].to(self.RUN_DEVICE)

                shape = w[x].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    shape = f"  {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f"  {str(shape[0]).rjust(5)}      "
                if block_id == 0:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    print(x.ljust(32), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device, shape)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)
        print(f'\nn_layer {args.n_layer} n_embd {args.n_embd} ctx_len {args.ctx_len}')

        keys = list(w.keys()) # store weights in self.w
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        with torch.no_grad(): # precompute embedding
            try:
                x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
            except:
                x = F.layer_norm(self.w.emb.weight.float(), (self.args.n_embd,), weight=self.w.blocks[0].ln0.weight.float(), bias=self.w.blocks[0].ln0.bias.float())
            self.w.emb.weight = x.to(dtype=self.FLOAT_MODE)

        self.eval()
        gc.collect()
        if 'cuda' in args.RUN_DEVICE:
            torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = torch.cat((state[:, 5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(1), x[:, :-1,:]), dim=1)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[:, 5*i+0] = x[:, -1,:].float()

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = torch.cat((state[:, 5*i+1].to(dtype=self.FLOAT_MODE).unsqueeze(1), x[:, :-1,:]), dim=1)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[:, 5*i+1] = x[:, -1,:].float()

        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[:, 5*i+2]
        bb = state[:, 5*i+3]
        pp = state[:, 5*i+4]
        T = x.shape[1]
        for t in range(T):
            ww = time_first + k[:, t]
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v[:, t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = torch.maximum(ww, k[:, t])
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k[:, t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[:, t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[:, 5*i+2] = e1 * aa + e2 * v[:, t]
                state[:, 5*i+3] = e1 * bb + e2
                state[:, 5*i+4] = p
            xx[:, t] = (a / b).to(dtype=self.FLOAT_MODE)
        return (r * xx) @ ow

    def forward(self, tokens, state):
        """
        Modified to handle batch input. It only returns the state. All parts of a batch must have the same number of tokens.
        """
        with torch.no_grad():
            w = self.w
            args = self.args
            x = w.emb.weight[torch.tensor(tokens)]
            if 'cuda' in self.RUN_DEVICE:
                x = x.to(self.RUN_DEVICE)

            if state == None:
                state = torch.zeros(len(tokens), args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[:, 5*i+4] -= 1e30

            SA = self.SA_seq
            FF = self.FF_seq

            for i in range(args.n_layer):
                ww = w.blocks[i].att
                x = x + SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn
                x = x + FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)

            return state