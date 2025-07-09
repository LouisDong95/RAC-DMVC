import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L2norm = nn.functional.normalize


class BaseModel(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(BaseModel, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes

        self.online_encoder = nn.ModuleList(
            [FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)]
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList(
            [MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)]
        )

        self.cl = ContrastiveLoss(temperature)
        self.ncl = DenoiseContrastiveLoss(temperature)
        self.dis = DistillLoss()
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

    def forward(self, *args, **kwargs):
        return self.forward_impl(*args, **kwargs)

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(
                self.online_encoder[i].parameters(), self.target_encoder[i].parameters()
            ):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask, realign=True):
        N = data[0].shape[0]
        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if realign:
            for i in range(1, self.n_views):
                bs = 1024
                tmp, z_tmp, z0 = (
                    L2norm(self.cross_view_decoder[i](z[i])),
                    torch.zeros(N, self.feature_dim[i]).cuda(),
                    z[0],
                )
                for j in range(int(np.ceil(z[i].shape[0] / bs))):
                    sim = z0[j * bs : (j + 1) * bs].mm(tmp.t())
                    idx = sim.argmax(1)
                    z_tmp[j * bs : (j + 1) * bs] = z[i][idx]
                z[i] = z_tmp

        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z


class DivideModel(BaseModel):
    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.0)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G

    def forward_impl(self, data, momentum, warm_up, singular_thresh, pseudo_centers):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp = torch.eye(z[0].shape[0]).cuda()
            mp = [mp, mp]
        else:
            mp = [self.kernel_affinity(z_t[i]) for i in range(self.n_views)]
        l_inter = (self.cl(p[0], z_t[1], mp[1]) + self.cl(p[1], z_t[0], mp[0])) / 2
        l_intra = (self.cl(z[0], z_t[0], mp[0]) + self.cl(z[1], z_t[1], mp[1])) / 2

        l_dist = torch.tensor(0.0).cuda()
        loss = {'l_intra': l_intra, 'l_inter': l_inter, 'l_dist': l_dist}
        return loss


class CandyModel(BaseModel):
    @torch.no_grad()
    def robust_affinity(self, p, z, t=0.07):
        G_intra, G_inter = [], []
        p = [L2norm(p[i]) for i in range(len(p))]
        z = [L2norm(z[i]) for i in range(len(z))]
        for i in range(len(p)):
            for j in range(len(z)):
                if i == j:
                    G = (2 - 2 * (z[i] @ z[j].t())).clamp(min=0.0)
                    G = torch.exp(-G / t)

                    G.fill_diagonal_(1.0)
                    G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                    G_intra.append(G)
                else:
                    G = (2 - 2 * (p[i] @ z[j].t())).clamp(min=0.0)
                    G = torch.exp(-G / t)

                    G[torch.eye(G.shape[0]) > 0] = (G[torch.eye(G.shape[0]) > 0] / G.diag().max().clamp_min(1e-7).detach())
                    G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                    G_inter.append(G)

        return G_intra, G_inter

    def forward_impl(self, data, momentum, warm_up, singular_thresh, pseudo_centers):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp_intra = torch.eye(z[0].shape[0]).cuda()
            mp_intra = [mp_intra, mp_intra]
            mp_inter = mp_intra
        else:
            mp_intra, mp_inter = self.robust_affinity(p, z_t)
        cc_loss, id_loss = 0.0, 0.0
        for i in range(self.n_views):
            for j in range(self.n_views):
                if i == j:
                    id_loss += self.ncl(
                        z[i],
                        z_t[i],
                        mp_intra[i].mm(mp_intra[j].t()),
                        association=(i, j),
                        singular_thresh=singular_thresh,
                    )
                else:
                    cc_loss += self.ncl(
                        p[i],
                        z_t[j],
                        mp_inter[i].mm(mp_intra[j].t())
                        + 0.2
                        * torch.eye(mp_inter[i].shape[0], device=mp_inter[i].device),
                        association=(i, j),
                        singular_thresh=singular_thresh,
                    )

        cc_loss, id_loss = cc_loss / self.n_views, id_loss / self.n_views

        if warm_up:
            mp_intra, mp_inter = self.robust_affinity(p, z_t)

        self.G_intra = mp_intra
        self.G_inter = mp_inter

        l_dist = torch.tensor(0.0).cuda()
        loss = {'l_intra': cc_loss, 'l_inter': id_loss, 'l_dist': l_dist}
        return loss


class NoisyModel(BaseModel):
    @torch.no_grad()
    def robust_affinity(self, p, z, t=0.07):
        G_intra, G_inter = [], []
        p = [L2norm(p[i]) for i in range(len(p))]
        z = [L2norm(z[i]) for i in range(len(z))]
        for i in range(len(p)):
            for j in range(len(z)):
                if i == j:
                    G = (2 - 2 * (z[i] @ z[j].t())).clamp(min=0.0)
                    G = torch.exp(-G / t)

                    G.fill_diagonal_(1.0)
                    G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                    G_intra.append(G)
                else:
                    G = (2 - 2 * (p[i] @ z[j].t())).clamp(min=0.0)
                    G = torch.exp(-G / t)

                    G.fill_diagonal_(1.0)
                    G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
                    G_inter.append(G)
        return G_intra, G_inter


    def forward_impl(self, data, momentum, warm_up, singular_thresh, pseudo_centers):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp_intra = torch.eye(z[0].shape[0]).cuda()
            mp_intra = [mp_intra, mp_intra]
            mp_inter = mp_intra
            l_dist = torch.tensor(0.0).cuda()
        else:
            mp_intra, mp_inter = self.robust_affinity(p, z_t)
            # l_dist = self.dis(p[0], p[1], pseudo_centers)
            l_dist = torch.tensor(0.0).cuda()

        l_intra = (self.cl.noisy_contrastive(z[0], z_t[0], mp_intra[0]) + self.cl.noisy_contrastive(z[1], z_t[1], mp_intra[1])) / 2
        l_inter = (self.cl.noisy_contrastive(p[0], z_t[1], mp_inter[0]) + self.cl.noisy_contrastive(p[1], z_t[0], mp_inter[1])) / 2

        loss = {'l_intra': l_intra, 'l_inter': l_inter, 'l_dist': l_dist}
        return loss

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        # prediction
        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        # # high
        # for i in range(self.n_views):
        #     G_local = self.kernel(z[i][mask[:, i]], z[i][mask[:, i]], flag=True)
        #     G_cross = self.kernel(z[1-i][~mask[:, i]], z[i][mask[:, i]], flag=False)
        #     z[i][~mask[:, i]] = G_cross @ G_local @ self.cross_view_decoder[1-i](z[1-i][mask[:, i]])

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z

    @torch.no_grad()
    def kernel(self, p, z, t=0.07, flag=False):
        p = L2norm(p)
        z = L2norm(z)
        G = (2 - 2 * (p @ z.t())).clamp(min=0.0)
        G = torch.exp(-G / t)
        if flag:
            G.fill_diagonal_(1.0)
        G = G / G.sum(1, keepdim=True).clamp_min(1e-7)
        return G



class FCN(nn.Module):
    def __init__(
        self,
        dim_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_out=0.0,
        norm_last_layer=True,
    ):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), act_layer(), nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.sum() / N
        # loss = nll_loss.mean()
        return loss

    def noisy_contrastive(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()

        logits = torch.matmul(x_q, x_k.T) / self.temperature
        logit_pos = (logits * mask_pos).sum(dim=1, keepdim=True)
        mask_neg = 1.0 - mask_pos

        exp_logits = torch.exp(logits) * mask_neg
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)

        loss = -torch.log(torch.exp(logit_pos) /  denom)
        return loss.mean()


class DenoiseContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        x_q,
        x_k,
        mask_pos=None,
        singular_thresh=0.2,
        association=None,
    ):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        loss = denoise_contrastive_loss(
            x_q, x_k, mask_pos, self.temperature, association, singular_thresh
        )

        return loss/N


def denoise_contrastive_loss(
    query,
    key,
    mask_pos,
    temperature,
    association,
    singular_thresh,
):
    similarity = torch.div(torch.matmul(query, key.T), temperature).softmax(1)

    logp = -similarity.log()

    enable_denoise = association[0] != association[1]

    L = mask_pos

    if enable_denoise:
        U, S, Vh = torch.linalg.svd(L)
        masked_out = S < singular_thresh
        S[masked_out] = 0
        L = U @ torch.diag(S) @ Vh

    L = L / L.sum(dim=1, keepdim=True).clamp_min(1e-7)

    nll_loss = L * logp

    loss = nll_loss.sum()
    return loss

class DistillLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, p_s, p_t, center):
        center = F.normalize(center, dim=-1)
        p_s = F.normalize(p_s, dim=-1)
        p_t = F.normalize(p_t, dim=-1)
        logits_s = p_s@center.t()/self.temperature
        logits_t = p_t@center.t()/self.temperature
        probs_s = F.softmax(logits_s, dim=-1).mean(0)
        probs_t = F.softmax(logits_t, dim=-1).mean(0)
        loss_uniform = (torch.sum(probs_s * torch.log(probs_s + 1e-8)) + torch.sum(probs_t * torch.log(probs_t + 1e-8)))/2

        p_s = F.log_softmax(p_s@center.t()/self.temperature, dim=-1)
        p_t = F.log_softmax(p_t@center.t()/self.temperature, dim=-1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean', log_target=True) * self.temperature ** 2
        return loss_uniform
