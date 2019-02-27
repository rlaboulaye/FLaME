import math
import copy
import json

import numpy as np
import torch
from torch import nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):

    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nf, nx)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w.t())
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class MLP(nn.Module):

    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg['n_embd']
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(cfg['resid_pdrop'])

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class LayerNorm(nn.Module):

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Attention(nn.Module):

    def __init__(self, embed_dim, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = embed_dim  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg['n_head'] == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg['n_head']
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, embed_dim)
        self.c_proj = Conv1D(n_state, 1, embed_dim)
        self.attn_dropout = nn.Dropout(cfg['attn_pdrop'])
        self.resid_dropout = nn.Dropout(cfg['resid_pdrop'])

    def _attn(self, query, key, value):
        w = torch.matmul(query, key)
        if self.scale:
            w = w / math.sqrt(value.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implementation method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, value)

    def merge_heads(self, heads):
        heads = heads.permute(0, 2, 1, 3).contiguous()
        new_x_shape = heads.size()[:-2] + (heads.size(-2) * heads.size(-1),)
        return heads.view(*new_x_shape)  # in Tensorflow implementation: fct merge_states

    def split_heads(self, head, is_key=False):
        new_head_shape = head.size()[:-1] + (self.n_head, head.size(-1) // self.n_head)
        head = head.view(*new_head_shape)  # in Tensorflow implementation: fct split_states
        if is_key:
            return head.permute(0, 2, 3, 1)
        else:
            return head.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class Block(nn.Module):

    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg['n_embd']
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class Transformer(nn.Module):

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg['n_embd'])
        self.drop = nn.Dropout(cfg['embd_pdrop'])
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg['n_layer'])])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        # Combine batch size and number of documents
        x = x.view(-1, x.size(-2), x.size(-1))
        embedding = self.embed(x)
        # Add the position information to the input embeddings
        encoding = embedding.sum(dim=2)
        for block in self.h:
            encoding = block(encoding)
        return encoding


class LanguageModelHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg):
        super(LanguageModelHead, self).__init__()
        self.n_embd = cfg['n_embd']
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights

    def forward(self, h):
        lm_logits = self.decoder(h)
        return lm_logits


class SingleHeadModel(nn.Module):
    """ Transformer with language model head """

    def __init__(self, cfg, vocab=40990, sequence_dim=512):
        super(SingleHeadModel, self).__init__()
        self.transformer = Transformer(cfg, vocab=vocab, n_ctx=sequence_dim)
        self.lm_head = LanguageModelHead(self.transformer, cfg)

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        return lm_logits


def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12,
        n_embd=768, path='./params/original/'):
    import re
    # Load weights from TF model
    names = json.load(open(path + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
            (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
            init_params[0]
            ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
            init_params[0]
            ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            if name[-1] == 'w':
                ip = ip.T
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)
