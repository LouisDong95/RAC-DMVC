import numpy as np
import torch
import torch.nn.functional as F
import utils
from model import BaseModel
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import MetricLogger, SmoothedValue, adjust_learning_config, AverageMeter


def train_one_epoch(
    model: BaseModel,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    state_logger=None,
    args=None,
    pseudo_centers = None
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50

    data_loader = enumerate(data_loader_train)

    model.train(True)
    optimizer.zero_grad()

    pseudo_centers = torch.from_numpy(pseudo_centers).float().cuda()

    Intra_losses = AverageMeter()
    Inter_losses = AverageMeter()
    Dist_losses = AverageMeter()
    for data_iter_step, (ids, samples, mask, _) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            loss_dict = model(samples, mmt, epoch < args.start_rectify_epoch, args.singular_thresh, pseudo_centers)

        intra_loss = loss_dict['l_intra']
        inter_loss = loss_dict['l_inter']
        dist_loss = loss_dict['l_dist']
        total_loss = intra_loss + inter_loss + dist_loss

        Intra_losses.update(intra_loss.item(), len(ids))
        Inter_losses.update(inter_loss.item(), len(ids))
        Dist_losses.update(dist_loss.item(), len(ids))

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if args.print_this_epoch:
        log = {'lr': lr, 'Intra_loss': Intra_losses.avg, 'Inter_loss': Inter_losses.avg, 'Dist_loss': Dist_losses.avg}
        print(log)


def evaluate(
    model: BaseModel,
    data_loader_test: DataLoader,
    device: torch.device,
    args=None,
):
    model.eval()
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        for indexs, samples, mask, labels in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            features = model.extract_feature(samples, mask)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels

        results = {}
        # for i in range(args.n_views):
        #     features_i = F.normalize(features_all[i], dim=-1).cpu().numpy()
        #     kmeans_i = KMeans(n_clusters=args.n_classes, random_state=0).fit(features_i)
        #     nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_i.labels_)
        #     results[f"view_{i}"] = {"nmi": nmi, "ari": ari, "f": f, "acc": acc}
        #     print(results[f"view_{i}"])

        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        # features_cat = torch.sum(features_all, dim=0) / args.n_views
        features_cat = F.normalize(features_cat, dim=-1).cpu().numpy()
        kmeans = KMeans(n_clusters=args.n_classes, random_state=0).fit(features_cat)

        nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans.labels_)
        result = {"nmi": nmi, "ari": ari, "f": f, "acc": acc}
        # print('result_fusion',result)
    return result, kmeans.cluster_centers_
