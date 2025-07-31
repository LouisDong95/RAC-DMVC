import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class BaseModel(torch.nn.Module):
    def __init__(self, n_views, layer_dims, n_classes, drop_rate=0.5, args=None):
        super(BaseModel, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.sigma = args.sigma

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.online_decoder = nn.ModuleList([FCNDecoder(dim_layer=layer_dims[i][::-1], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])
        self.clustering_layer = nn.ModuleList([Classifier(n_classes, layer_dims[i][-1]) for i in range(n_views)])
        self._initialize_target_encoders()

        self.cl = ContrastiveLoss(args.con_temperature)
        self.ncl = DenoiseContrastiveLoss(args.con_temperature)
        self.dist = DistillLoss(args.dist_temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        self.cluster_centers = None

    def forward(self, *args, **kwargs):
        return self.forward_impl(*args, **kwargs)

    def _initialize_target_encoders(self):
        for online_encoder, target_encoder in zip(self.online_encoder, self.target_encoder):
            for param_q, param_k in zip(online_encoder.parameters(), target_encoder.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(
                self.online_encoder[i].parameters(), self.target_encoder[i].parameters()
            ):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask, realign=False):
        N = data[0].shape[0]
        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if realign:
            for i in range(1, self.n_views):
                bs = 1024
                tmp, z_tmp, z0 = (
                    F.normalize(self.cross_view_decoder[i](z[i])),
                    torch.zeros(N, self.feature_dim[i]).cuda(),
                    z[0],
                )
                for j in range(int(np.ceil(z[i].shape[0] / bs))):
                    sim = z0[j * bs : (j + 1) * bs].mm(tmp.t())
                    idx = sim.argmax(1)
                    z_tmp[j * bs : (j + 1) * bs] = z[i][idx]
                z[i] = z_tmp

        z = [F.normalize(z[i]) for i in range(self.n_views)]
        return z

    @torch.no_grad()
    def update_cluster_centers(self, data, labels):
        with torch.no_grad():
            target_features = [encoder(view) for encoder, view in zip(self.target_encoder, data)]
            # fused_features = F.normalize(torch.stack(target_features, dim=0).sum(dim=0)/2)
            fused_features = F.normalize(torch.cat(target_features, dim=1), dim=1)
            kmeans = KMeans(n_clusters=self.n_classes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(fused_features.cpu().numpy())

            cluster_centers = []
            cluster_labels_tensor = torch.tensor(cluster_labels, device=fused_features.device)
            for c in range(self.n_classes):
                mask = (cluster_labels_tensor == c)
                center = fused_features[mask].mean(dim=0)
                cluster_centers.append(center)
            self.cluster_centers = torch.stack(cluster_centers)

    # def initialize_clustering_layer(self):
    #     for clustering_layer in self.clustering_layer:
    #         clustering_layer.centers.data.copy_(self.cluster_centers)

class DivideModel(BaseModel):
    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = F.normalize(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.0)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G

    def forward_impl(self, data_ori, data_noise, warm_up, singular_thresh):
        z = [self.online_encoder[i](data_noise[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data_noise[i]) for i in range(self.n_views)]

        if warm_up:
            mp = torch.eye(z[0].shape[0]).cuda()
            mp = [mp, mp]
        else:
            mp = [self.kernel_affinity(z_t[i]) for i in range(self.n_views)]
        l_inter = (self.cl(p[0], z_t[1], mp[1]) + self.cl(p[1], z_t[0], mp[0])) / 2
        l_intra = (self.cl(z[0], z_t[0], mp[0]) + self.cl(z[1], z_t[1], mp[1])) / 2
        loss = {'l_rec': torch.tensor(0.0).cuda(), 'l_intra': l_intra, 'l_inter': l_inter, 'l_dist': torch.tensor(0.0).cuda()}
        return loss


class CandyModel(BaseModel):
    @torch.no_grad()
    def robust_affinity(self, p, z, t=0.07):
        G_intra, G_inter = [], []
        p = [F.normalize(p[i]) for i in range(len(p))]
        z = [F.normalize(z[i]) for i in range(len(z))]
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

    def forward_impl(self, data_ori, data_noise, warm_up, singular_thresh):
        z = [self.online_encoder[i](data_noise[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data_noise[i]) for i in range(self.n_views)]

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
        loss = {'l_rec': torch.tensor(0.0).cuda(), 'l_intra': cc_loss, 'l_inter': id_loss, 'l_dist': torch.tensor(0.0).cuda()}
        return loss


class NoisyModel(BaseModel):
    @torch.no_grad()
    def robust_affinity(self, p, z, t=0.07):
        G_intra, G_inter = [], []
        p = [F.normalize(p[i]) for i in range(len(p))]
        z = [F.normalize(z[i]) for i in range(len(z))]
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


    def forward_impl(self, data_ori, data_noise, warm_up, singular_thresh):
        z = [self.online_encoder[i](data_noise[i]) for i in range(self.n_views)]
        x_r = [self.online_decoder[1-i](z[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data_noise[i]) for i in range(self.n_views)]

        if warm_up:
            mp_intra = torch.eye(z[0].shape[0]).cuda()
            mp_intra = [mp_intra, mp_intra]
            mp_inter = mp_intra
            l_dist = torch.tensor(0.0).cuda()
        else:
            mp_intra, mp_inter = self.robust_affinity(p, z_t, self.sigma)
            logit_p = [self.clustering_layer[i](p[i]) for i in range(self.n_views)]
            l_dist = (self.dist(logit_p[0], z_t, self.cluster_centers) + self.dist(logit_p[1], z_t, self.cluster_centers))/2

        l_rec = (F.mse_loss(data_ori[0], x_r[1], reduction='mean') + F.mse_loss(data_ori[1], x_r[0], reduction='mean')) / 2
        # l_rec = (F.mse_loss(data[0], x_r[0], reduction='mean') + F.mse_loss(data[1], x_r[1], reduction='mean'))/2
        # l_intra = (self.cl(z[0], z_t[0]) + self.cl(z[1], z_t[1])) / 2
        # l_inter = (self.cl(p[0], z_t[1]) + self.cl(p[1], z_t[0])) / 2

        l_intra = (self.cl.noisy_contrastive(z[0], z_t[0], mp_intra[0]) + self.cl.noisy_contrastive(z[1], z_t[1], mp_intra[1])) / 2
        l_inter = (self.cl.noisy_contrastive(p[0], z_t[1], mp_inter[0]) + self.cl.noisy_contrastive(p[1], z_t[0], mp_inter[1])) / 2

        loss = {'l_rec': l_rec, 'l_intra': l_intra, 'l_inter': l_inter, 'l_dist': l_dist}
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

        alpha = 0.5
        for i in range(self.n_views):
            idx_obs = mask[:, i]
            idx_miss = ~idx_obs

            z_obs = z[i][idx_obs]
            z_cross_obs = self.cross_view_decoder[1 - i](z[1 - i][mask[:, 1 - i]])

            z_miss_query = self.cross_view_decoder[1 - i](z[1 - i][idx_miss])

            attn_cross = self.graph_attention(z_miss_query, z_cross_obs, z_cross_obs, self.sigma)
            attn_local = self.graph_attention(z_miss_query, z_obs, z_obs, self.sigma)

            z[i][idx_miss] = alpha * attn_cross + (1 - alpha) * attn_local
        # for i in range(self.n_views):
        #     sim = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]]) @ self.cluster_centers.T
        #     sim = torch.argmax(sim, dim=1)
        #     z[i][~mask[:, i]] = self.cluster_centers[sim]

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [F.normalize(z[i]) for i in range(self.n_views)]
        return z

    @torch.no_grad()
    def graph_attention(self, query, key, value, temperature=0.07):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attn_score = -(2- 2*(query @ key.T)) / temperature  # [N_missing, N_obs]
        attn = F.softmax(attn_score, dim=-1)  # attention weights
        out = attn @ value  # [N_missing, D]
        return out


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

class FCNDecoder(nn.Module):
    def __init__(
        self,
        dim_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_out=0.0,
        norm_last_layer=False
    ):
        super(FCNDecoder, self).__init__()
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

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

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

class Classifier(nn.Module):
    def __init__(self, n_clusters, feature_dim):
        super(Classifier, self).__init__()
        self.centers = nn.Parameter(torch.randn(n_clusters, feature_dim))

    def forward(self, p):
        features = F.normalize(p, dim=-1)
        centers = F.normalize(self.centers, dim=-1)
        logits = torch.mm(features, centers.t())
        return logits



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = F.normalize(x_q)
        x_k = F.normalize(x_k)
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
        x_q = F.normalize(x_q)
        x_k = F.normalize(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        mask_neg = 1.0 - mask_pos

        logits = torch.matmul(x_q, x_k.T) / self.temperature
        logit_pos = (logits * mask_pos).sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits) * mask_neg
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        loss = -torch.log(torch.exp(logit_pos) / denom)
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
        x_q = F.normalize(x_q)
        x_k = F.normalize(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        loss = denoise_contrastive_loss(
            x_q, x_k, mask_pos, self.temperature, association, singular_thresh
        )
        loss = loss.sum()/N
        return loss


def denoise_contrastive_loss(query,key,mask_pos,temperature,association,singular_thresh,):
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
    # nll_loss = nll_loss.mean()
    return nll_loss


class DistillLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits, z, centers):
        p = F.log_softmax(logits/self.temperature, dim=1)
        z = torch.cat(z, dim=1)
        z = F.normalize(z)
        centers = F.normalize(centers)
        q = F.softmax(torch.mm(z, centers.t())/self.temperature, dim=1)
        kl_loss = F.kl_div(p, q, reduction="batchmean")
        return kl_loss