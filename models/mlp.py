import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import register_model

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

@register_model('mlp')
class MLPModel(nn.Module):
    def __init__(self, alphabet_size, num_cls, hidden_dim=128, cls_expanded_simplex=False, cls_free_guidance=False, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        self.hidden_dim = hidden_dim
        self.cls_expanded_simplex = cls_expanded_simplex
        self.cls_free_guidance = cls_free_guidance

        self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_dim),nn.Linear(self.hidden_dim, self.hidden_dim))
        self.embedder = nn.Linear((1 if not self.cls_expanded_simplex else 2) * self.alphabet_size,  self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim if classifier else self.alphabet_size)
        )
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.num_cls))
        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden_dim)


    def forward(self, seq, t, cls=None):
        time_embed = self.time_embedder(t)
        # print("seq:", seq.shape, "embedder.in_features:", self.embedder.in_features)
        feat = self.embedder(seq)
        feat = feat + time_embed[:,None,:]
        if self.cls_free_guidance and not self.classifier:
            feat = feat + self.cls_embedder(cls)[:, None, :]
        feat = self.mlp(feat)
        if self.classifier:
            return self.cls_head(feat.mean(dim=1))
        else:
            return feat

# @register_model('mlp')
# class ThreeLayerMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(ThreeLayerMLP, self).__init__()
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# @register_model('mlp_dp')
# class ThreeLayerMLP_with_dropout(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super().__init__()
        
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim2, output_dim)
#         )

#     def forward(self, x):
#         return self.net(x)

# -----------------------------
# Simple MLP velocity model v_theta(x_t, t)
# enforce sum(v)=0 so the ODE preserves sum(x)=1
# -----------------------------
@register_model('v_mlp')
class VelocityMLP(nn.Module):
    def __init__(self, K=10, hidden=256, time_embed=64):
        super().__init__()
        self.K = K
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed),
            nn.SiLU(),
            nn.Linear(time_embed, time_embed),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(K + time_embed, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x, t):
        """
        x: [B,K] on simplex
        t: [B] in [0,1]
        """
        t = t.view(-1, 1)
        te = self.time_mlp(t)
        h = torch.cat([x, te], dim=-1)
        v = self.net(h)

        # project to sum(v)=0 so sum(x) stays 1 under dx/dt=v
        v = v - v.sum(dim=-1, keepdim=True) / self.K
        return v

class DVFMMLP(nn.Module):
    """
    model(x,t) -> (v_pred, alpha_theta)
    alpha_theta positive (Dirichlet concentration) for sampling x1_pred
    """
    def __init__(self, K=10, hidden=256, time_embed=64):
        super().__init__()
        self.K = K
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed), nn.SiLU(),
            nn.Linear(time_embed, time_embed), nn.SiLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(K + time_embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_v = nn.Linear(hidden, K)
        self.head_a = nn.Linear(hidden, K)

    def forward(self, x, t):
        t = t.view(-1, 1)
        te = self.time_mlp(t)
        h = self.backbone(torch.cat([x, te], dim=-1))
        v = project_tangent_simplex(self.head_v(h))
        alpha = F.softplus(self.head_a(h)) + 1e-4
        return v, alpha

def project_tangent_simplex(v):
    return v - v.mean(dim=-1, keepdim=True)

@register_model('dirichlet_mlp')
class DirichletModeClassifier(nn.Module):
    """
    For dirichlet FM classification:
      model(x_t, t) -> logits [B,M]
    """
    def __init__(self, K=10, M=5, hidden=256, time_embed=64):
        super().__init__()
        self.K = K
        self.M = M
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed), nn.SiLU(),
            nn.Linear(time_embed, time_embed), nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(K + time_embed, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, M),
        )

    def forward(self, x, t):
        t = t.view(-1, 1)
        te = self.time_mlp(t)
        return self.net(torch.cat([x, te], dim=-1))