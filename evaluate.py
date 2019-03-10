import torch


class Evaluator:
    def __init__(self, lm_criterion, r_coef=1000., d_coef=1000.):
        self.lm_criterion = lm_criterion
        self.r_coef = r_coef
        self.d_coef = d_coef

    def compute_flame_loss(self, model, x, m, z, logdet, lm_logits):
        nll = self.compute_nll_loss(model, m, z, logdet)
        reconstruction_loss = self.compute_lm_loss(x, m, lm_logits)
        distance_loss = self.compute_distance_loss(m, z)
        loss = nll + self.r_coef * reconstruction_loss + self.d_coef * distance_loss
        return loss, nll, reconstruction_loss, distance_loss

    def compute_nll_loss(self, model, m, z, logdet):
        nll = -1. * (model.prior.log_prob(z) + logdet)
        m_flat = m[:, 1:].contiguous().view(-1)
        nll = nll * m_flat
        return nll.mean()

    def compute_distance_loss(self, m, z):
        z = z.view(m.shape + (-1,))
        mu = z.mean(dim=-2)
        d = z - mu.repeat(1, z.shape[-2]).view(z.shape)
        # Temporarily simplified - this is not the mahalanobis distance
        distances = d.pow(2).sum(dim=-1).sqrt()[:, :-1].contiguous().view(-1)
        # cov = d.transpose(-1,-2).matmul(d) / z.shape[-2]
        # distances = (d.matmul(cov.inverse()) * d).sum(dim=-1).sqrt()[:, :-1].contiguous().view(-1)
        m_flat = m[:, 1:].contiguous().view(-1)
        distance_losses = distances * m_flat
        return distance_losses.mean()

    def compute_lm_loss(self, X, M, lm_logits):
        X_shifted = X[:, 1:, 0].contiguous().view(-1)
        M = M.view(-1, M.size(-1))
        # Truncated Language modeling logits (we remove the last token)
        lm_logits_trunc = lm_logits[:, :-1].contiguous().view(X.shape[0] * (X.shape[1] - 1), -1)
        lm_losses = self.lm_criterion(lm_logits_trunc, X_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        return lm_losses.mean()
