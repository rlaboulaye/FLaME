import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .transformer_models import Transformer


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
        self.initialized = False

    def initialize(self, h):
        with torch.no_grad():
            shift = h.mean(dim=0)
            scale_log = h.std(dim=0).log()
            self.shift.data.copy_(shift.data)
            self.scale_log.data.copy_(scale_log.data)
        self.initialized = True

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

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers - 1)])
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = nn.ReLU()

    def forward(self, h):
        for layer in self.layers:
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
        h1, h2 = self.split(h)
        shift, scale_log = self.split(self.f(h1))
        scale = scale_log.exp()
        h2 = h2 * scale + shift
        logdet += scale_log.sum(dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return h, logdet

    def reverse_flow(self, h, logdet):
        h1, h2 = self.split(h)
        shift, scale_log = self.split(self.f(h1))
        scale = scale_log.exp()
        h2 = (h2 - shift) / scale
        logdet -= scale_log.sum(dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        h, logdet = self.permutation(h, logdet=logdet, reverse=True)
        h, logdet = self.actnorm(h, logdet=logdet, reverse=True)
        return h, logdet

    def split(self, h):
        embedding_dim = h.shape[-1]
        return h.split(embedding_dim // 2, dim=-1)


class FLaME(nn.Module):

    def __init__(self, cfg, n_pre=2, n_post=1):
        super(FLaME, self).__init__()
        embedding_dim = cfg['n_embd']
        self.prior = MultivariateNormal(torch.zeros(embedding_dim), torch.eye(embedding_dim))
        self.pre_steps = nn.ModuleList([FlowStep(embedding_dim) for i in range(n_pre)])
        self.post_steps = nn.ModuleList([FlowStep(embedding_dim) for i in range(n_post)])
        self.language_model = Transformer(cfg)
        self.vocab_projection = nn.Linear(*self.language_model.embed.weight.shape, bias=False)
        self.vocab_projection.weight = self.language_model.embed.weight

    def forward(self, h, y, reverse=False):
        # y = self.language_model(y)[:, -1]
        logdet = torch.zeros(h.shape[0], dtype=torch.float32)
        if not reverse:
            return self.encode(h, y, logdet)
        else:
            return self.decode(h, y, logdet)

    def encode(self, x, y, logdet):
        h = x
        for step in self.pre_steps:
            h, logdet = step(h, logdet)
        # incorporate y
        for step in self.post_steps:
            h, logdet = step(h, logdet)
        return h, logdet

    def decode(self, z, y, logdet):
        h = z
        for step in reversed(self.post_steps):
            h, logdet = step(h, logdet, reverse=True)
        # incorporate y
        for step in reversed(self.pre_steps):
            h, logdet = step(h, logdet, reverse=True)
        return h, logdet

    def project(self, x):
        return self.vocab_projection(x)
