import math
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions import Normal


class SquashedNormal(Normal):
    def __init__(self,
                 loc: torch.Tensor,
                 scale: torch.Tensor,
                 ):
        super().__init__(loc, scale)
        self.epsilon = np.finfo(np.float32).eps.item()

    def log_prob(self,
                 value: torch.Tensor,
                 pre_tanh_value: bool=False,
                 ):
        if not pre_tanh_value:
            pre_value = torch.clip(value, -1.0+self.epsilon, 1.0-self.epsilon)
            pre_value = 0.5 * (pre_value.log1p() - (-pre_value).log1p())
        else:
            pre_value = value
            value = torch.tanh(pre_value)
        return super().log_prob(pre_value) - 2*(math.log(2.0) - pre_value - torch.nn.functional.softplus(-2 * pre_value))

    def sample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().sample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)

    def rsample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().rsample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)

    def entropy(self):
        return super().entropy()

    @property
    def tanh_mean(self):
        return torch.tanh(self.mean)
    
class GmmScore(nn.Module):
    def __init__(self, n_modes, dim):
        super().__init__()
        self.n_modes = n_modes 
        self.dim = dim

        # initialise means and weights
        self.loc = nn.Parameter(torch.randn(self.n_modes, self.dim))
        self.log_weights = nn.Parameter(torch.zeros(self.n_modes))

        # build lower triangular matrices for covariances 
        self.n_tril = self.dim * (self.dim + 1) // 2
        self.tril_raw = nn.Parameter(torch.zeros(self.n_modes, self.n_tril))

    def _build_cholesky(self):
        """
        Assemble lower triangular L_k from raw params.
        Diagonal entries passed through softplus to ensure positivity.
        """
        K, D = self.n_modes, self.dim
        L = torch.zeros(K, D, D, device=self.tril_raw.device)

        # Indices for lower triangular entries
        tril_idx = torch.tril_indices(D, D, device=self.tril_raw.device)  # (2, n_tril)
        diag_mask = tril_idx[0] == tril_idx[1]                            # which entries are diagonal

        entries = self.tril_raw.clone()
        entries[:, diag_mask] = F.softplus(self.tril_raw[:, diag_mask])   # (K, n_tril) positive diag

        L[:, tril_idx[0], tril_idx[1]] = entries                          # (K, D, D)
        return L
    
    def forward(self, x):
        """
        Compute score ∇_x log p(x).

        x: (N, D)
        returns: (N, D)
        """
        weights = F.softmax(self.log_weights, dim=0)   # (K,)  normalized
        L       = self._build_cholesky()               # (K, D, D)

        return self.gmm_score_analytical(x, weights, self.loc, L)
    
    def gmm_score_analytical(self, x, weights, means, chol_covs):
        D = self.dim

        diff      = x[:, None, :] - means[None, :, :]              # (N, K, D)
        diff_T    = diff.permute(1, 2, 0)                          # (K, D, N)

        v         = torch.linalg.solve_triangular(
                        chol_covs, diff_T, upper=False)            # (K, D, N)
        prec_diff = torch.linalg.solve_triangular(
                        chol_covs.mT, v, upper=True
                    ).permute(2, 0, 1)                             # (N, K, D)

        log_det   = 2 * chol_covs.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (K,)
        mahal     = 0.5 * (diff * prec_diff).sum(-1)               # (N, K)

        log_densities = (
            -0.5 * D * torch.log(torch.tensor(2 * torch.pi))
            - 0.5 * log_det
            - mahal
        )                                                          # (N, K)

        log_r = weights.log() + log_densities
        r     = F.softmax(log_r, dim=1)                            # (N, K)
        score = -(r.unsqueeze(-1) * prec_diff).sum(1)              # (N, D)
        return score