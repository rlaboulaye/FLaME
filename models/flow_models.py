import numpy as np
import torch
from torch import nn
from torch.nn import Softmax
from torch.distributions.multivariate_normal import MultivariateNormal

from .transformer_models import Transformer


def split(h):
    embedding_dim = h.shape[-1]
    return h.split(embedding_dim // 2, dim=-1)


class Permutation(nn.Module):

    def __init__(self, embedding_dim):
        super(Permutation, self).__init__()
        W_init = np.linalg.qr(np.random.randn(embedding_dim, embedding_dim))[0].astype(np.float32)
        self.W = nn.Parameter(torch.Tensor(W_init))

    def forward(self, h, logdet, reverse=False):
        if not reverse:
            h = h.matmul(self.W)
            logdet += self.W.transpose(0, 1).slogdet()[1]
        else:
            h = h.matmul(self.W.inverse())
            logdet -= self.W.transpose(0, 1).slogdet()[1]
        return h, logdet


class ActNorm(nn.Module):

    def __init__(self, embedding_dim):
        super(ActNorm, self).__init__()
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
        self.scale_log = nn.Parameter(torch.zeros(embedding_dim))
        self.register_buffer('initialized', torch.zeros(1).byte())

    def initialize(self, h):
        with torch.no_grad():
            shift = h.mean(dim=0)
            scale_log = h.std(dim=0).log()
            self.shift.data.copy_(shift.data)
            self.scale_log.data.copy_(scale_log.data)
        self.initialized = torch.ones(1).byte()

    def forward(self, h, logdet, reverse=False):
        if not self.initialized:
            self.initialize(h)
        if not reverse:
            h = (h - self.shift) * self.scale_log.exp()
            logdet += self.scale_log.sum()
        else:
            h = (h / self.scale_log.exp()) + self.shift
            logdet -= self.scale_log.sum()
        return h, logdet


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, final_activation=True):
        super(MLP, self).__init__()
        if n_layers == 0:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers - 1)])
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = nn.ReLU()
        self.final_activation = final_activation

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and not self.final_activation:
                h = layer(h)
            else:
                h = self.activation(layer(h))
        return h


class FlowStep(nn.Module):

    def __init__(self, embedding_dim):
        super(FlowStep, self).__init__()
        self.actnorm = ActNorm(embedding_dim)
        self.permutation = Permutation(embedding_dim)
        self.f = MLP(embedding_dim // 2, embedding_dim, embedding_dim)

    def forward(self, h, logdet, reverse=False):
        if not reverse:
            return self.flow(h, logdet)
        else:
            return self.reverse_flow(h, logdet)

    def flow(self, h, logdet):
        h, logdet = self.actnorm(h, logdet=logdet, reverse=False)
        h, logdet = self.permutation(h, logdet=logdet, reverse=False)
        h1, h2 = split(h)
        shift, scale_raw = split(self.f(h1))
        scale = (scale_raw + 2.).sigmoid()
        h2 = (h2 + shift) * scale
        logdet += scale.log().sum(dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return h, logdet

    def reverse_flow(self, h, logdet):
        h1, h2 = split(h)
        shift, scale_raw = split(self.f(h1))
        scale = (scale_raw + 2.).sigmoid()
        h2 = (h2 / scale) - shift
        logdet -= scale.log().sum(dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        h, logdet = self.permutation(h, logdet=logdet, reverse=True)
        h, logdet = self.actnorm(h, logdet=logdet, reverse=True)
        return h, logdet


class ConditionalFlowNet(nn.Module):

    def __init__(self, cfg):
        super(ConditionalFlowNet, self).__init__()
        embedding_dim = cfg['n_embd']
        n_pre_layer = cfg['n_pre_layer']
        n_post_layer = cfg['n_post_layer']
        self.pre_steps = nn.ModuleList([FlowStep(embedding_dim) for i in range(n_pre_layer)])
        self.post_steps = nn.ModuleList([FlowStep(embedding_dim) for i in range(n_post_layer)])
        self.f = MLP(embedding_dim, embedding_dim * 2, embedding_dim * 2)

    def forward(self, h, y, reverse=False):
        logdet = torch.zeros(h.shape[0], dtype=torch.float32, device=next(self.parameters()).device)
        if not reverse:
            return self.encode(h, y, logdet)
        else:
            return self.decode(h, y, logdet)

    def encode(self, x, y, logdet):
        h = x
        for step in self.pre_steps:
            h, logdet = step(h, logdet)

        shift, scale_raw = split(self.f(y))
        scale = (scale_raw + 2.).sigmoid()
        h = (h + shift) * scale
        logdet += scale.log().sum(dim=-1)

        for step in self.post_steps:
            h, logdet = step(h, logdet)
        return h, logdet

    def decode(self, z, y, logdet):
        h = z
        for step in reversed(self.post_steps):
            h, logdet = step(h, logdet, reverse=True)

        shift, scale_raw = split(self.f(y))
        scale = (scale_raw + 2.).sigmoid()
        h = (h / scale) - shift
        logdet -= scale.log().sum(dim=-1)

        for step in reversed(self.pre_steps):
            h, logdet = step(h, logdet, reverse=True)
        return h, logdet

    def project(self, x):
        return self.vocab_projection(x)


class FLaME(nn.Module):

    def __init__(self, cfg, vocab=40990):
        super(FLaME, self).__init__()
        self.embedding_dim = cfg['n_embd']
        self.conditional_flow = ConditionalFlowNet(cfg)
        self.language_model = Transformer(cfg, vocab)
        self.vocab_projection = MLP(self.language_model.embed.weight.shape[1],
            self.language_model.embed.weight.shape[1] * 2,
            self.language_model.embed.weight.shape[0],
            n_layers=cfg['n_projection_layer'], final_activation=False)
        self.softmax = Softmax(dim=-1)
        self.prior = MultivariateNormal(torch.zeros(self.embedding_dim), torch.eye(self.embedding_dim))

    def to(self, device):
        super(FLaME, self).to(device)
        self.prior = MultivariateNormal(torch.zeros(self.embedding_dim, device=device), torch.eye(self.embedding_dim, device=device))

    def forward(self, x):
        x_embd = self.language_model.embed(x[:, :, 0].contiguous()).clone().detach()
        y = self.language_model(x)
        x_embd_flat = x_embd[:, 1:].contiguous().view(-1, x_embd.shape[-1])
        y_flat = y[:, :-1].contiguous().view(-1, y.shape[-1])
        z, logdet = self.conditional_flow(x_embd_flat, y_flat)
        lm_logits = self.vocab_projection(x_embd)
        return z, logdet, lm_logits

    def generate(self, x, position_token, z=None, max_length=512):
        if z is None:
            z = self.prior.sample((x.shape[0],))
        with torch.no_grad():
            for i in range(max_length - x.shape[1]):
                y = self.language_model(x)
                h, _ = self.conditional_flow(z, y[:, -1], reverse=True)
                logits = self.vocab_projection(h)
                x_new = torch.zeros((x.shape[0], 1, 2), dtype=torch.int64, device=next(self.parameters()).device)
                x_new[:, :, 0] = torch.multinomial(self.softmax(logits), 1)
                x_new[:, :, 1] = position_token + x.shape[1]
                x = torch.cat([x, x_new], dim=1)
        return x
