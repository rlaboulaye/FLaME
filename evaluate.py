import torch


class Evaluator:
    def __init__(self, lm_criterion):
        self.lm_criterion = lm_criterion

    def compute_loss(self, X, M, lm_logits):
        X_shifted = X[:, 1:, 0].contiguous().view(-1)
        M = M.view(-1, M.size(-1))
        lm_losses = self.lm_criterion(lm_logits, X_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        return lm_losses.sum()
