import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from .conv_layers import ResidualBlock, RefineBlock, get_act
from .normalization import get_normalization
from .._base import register_model


def _compute_cond_module(module, x):
    for m in module:
        x = m(x)
    return x


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

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

@register_model('cnn')
class CNNModel(nn.Module):
    def __init__(self,
            dim: int,
            k: int,
            hidden: int,
            mode: str,
            num_cls: int,
            depth: int,
            dropout: float,
            prior_pseudocount: float,
            cls_expanded_simplex: bool,
            clean_data: bool = False,
            classifier: bool = False,
            classifier_free_guidance: bool = False,
            **_,
        ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.hidden = hidden
        self.mode = mode
        self.depth = depth
        self.dropout = dropout
        self.prior_pseudocount = prior_pseudocount
        self.cls_expanded_simplex = cls_expanded_simplex
        self.classifier = classifier
        self.cls_free_guidance = classifier_free_guidance
        self.clean_data = clean_data
        self.num_cls = num_cls

        if self.clean_data:
            self.linear = nn.Embedding(self.dim, embedding_dim=hidden)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or not classifier and (self.mode == 'dirichlet' or self.mode == 'riemannian')
            inp_size = self.k * (2 if expanded_simplex_input else 1)
            # print("input size is %d" % (inp_size))
            if (self.mode == 'ardm' or self.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, self.hidden, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden), nn.Linear(self.hidden,self.hidden))

        self.num_layers = 5 * self.depth
        self.convs = [nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(self.hidden, self.hidden, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(self.depth)])
        self.time_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(self.hidden, self.hidden, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv1d(self.hidden, self.hidden if classifier else self.k, kernel_size=1))
        self.dropout = nn.Dropout(self.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(self.hidden, self.hidden),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden, self.num_cls))
        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden)
            self.cls_layers = nn.ModuleList([Dense(self.hidden, self.hidden) for _ in range(self.num_layers)])

    def forward(self, x, t: Tensor, cls = None, return_embedding=False):
        # print(f"input x shape = {x.shape}")
        if self.clean_data:
            feat = self.linear(x)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = x.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            # print(f"h shape before time add = {h.shape}")
            # tmp = self.time_layers[i](time_emb)            
            # print(f"projected time shape = {tmp.shape}")
            # print(f"projected time unsqueezed = {tmp[:, :, None].shape}")
            if not self.clean_data:
                # print(f"time_emb shape = {time_emb.shape}")
                time_feat = self.time_layers[i](time_emb).squeeze(1)
                # h = h + self.time_layers[i](time_emb)[:, :, None]
                h = h + time_feat[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                # print(f"cls_emb shape = {cls_emb.shape}")
                cls_feat = self.cls_layers[i](cls_emb).squeeze(1)
                # h = h + self.cls_layers[i](cls_emb)[:, :, None]
                h = h + cls_feat[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat

class ConvNet(nn.Module):
    def __init__(self, in_channels=2, normalization='InstanceNorm++', ngf=128, nonlinearity='elu'):
        super().__init__()
        self.in_channels = in_channels
        self.normalization = normalization
        self.ngf = ngf
        self.nonlinearity = nonlinearity

        self.norm = get_normalization(normalization, conditional=False)
        self.act = act = get_act(nonlinearity)

        self.begin_conv = nn.Conv2d(in_channels, ngf, 3, stride=1, padding=1)
        self.fc_t1 = nn.Linear(ngf, ngf)
        self.normalizer = self.norm(ngf, 0)
        self.end_conv = nn.Conv2d(ngf, in_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm)
        ])
        self.fc_t2 = nn.Linear(ngf, ngf)

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm)
        ])
        self.fc_t3 = nn.Linear(ngf, 2 * ngf)

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)
        ])
        self.fc_t4 = nn.Linear(ngf, 2 * ngf)

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, adjust_padding=False, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)
        ])
        self.fc_t5 = nn.Linear(ngf, 2 * ngf)

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def forward(self, x, t):
        x = x.permute(0, 3, 1, 2)
        t_embed = get_time_embedding(t, self.ngf)
        output = self.begin_conv(2 * x - 1) + self.fc_t1(t_embed)[..., None, None]

        layer1 = _compute_cond_module(self.res1, output) + self.fc_t2(t_embed)[..., None, None]
        layer2 = _compute_cond_module(self.res2, layer1) + self.fc_t3(t_embed)[..., None, None]
        layer3 = _compute_cond_module(self.res3, layer2) + self.fc_t4(t_embed)[..., None, None]
        layer4 = _compute_cond_module(self.res4, layer3) + self.fc_t5(t_embed)[..., None, None]

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        return output.permute(0, 2, 3, 1)
