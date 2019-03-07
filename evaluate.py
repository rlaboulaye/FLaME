import torch

from sinkhorn import SinkhornDistance


class Evaluator:
    def __init__(self, lm_criterion):
        self.lm_criterion = lm_criterion

    def compute_loss(self, X, M, lm_logits):
        X_shifted = X[:, 1:, 0].contiguous().view(-1)
        M = M.view(-1, M.size(-1))
        # Truncated Language modeling logits (we remove the last token)
        lm_logits_trunc = lm_logits[:, :-1].contiguous().view(X.shape[0] * (X.shape[1] - 1), -1)
        if isinstance(self.lm_criterion, SinkhornDistance):
            target_one_hot = torch.zeros(lm_logits_trunc.shape).scatter_(-1, X_shifted.unsqueeze(-1).cpu(), 1.)
            # loss currently takes into account logits that should be masked
            lm_loss, _, _ = self.lm_criterion(lm_logits_trunc.float().cpu(), target_one_hot.float().cpu())
        else:
            lm_losses = self.lm_criterion(lm_logits_trunc, X_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
            lm_loss = lm_losses.mean()
        return lm_loss

# class Evaluator:
#     def __init__(self, lm_criterion):
#         self.lm_criterion = lm_criterion

#     def compute_loss(self, X, M, lm_logits):
#         X_shifted = X[:, 1:, 0].contiguous().view(-1)
#         M = M.view(-1, M.size(-1))
#         # Truncated Language modeling logits (we remove the last token)
#         lm_logits_trunc = lm_logits[:, :-1].contiguous().view(X.shape[0] * (X.shape[1] - 1), -1)
#         lm_losses = self.lm_criterion(lm_logits_trunc, X_shifted)
#         lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
#         lm_losses = lm_losses * M[:, 1:]
#         lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
#         return lm_losses.sum()
