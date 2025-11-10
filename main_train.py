import argparse
import copy
import datetime
import os
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import utils
import yaml
from dataset_loader import load_dataset, IncompleteDatasetSampler
from engine_train import train_one_epoch, evaluate
from model import DivideModel, CandyModel, NoisyModel
from torch.utils import data
import csv

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training")

    # config path
    parser.add_argument("--config_file", type=str, default=None)

    # backbone parameters
    parser.add_argument("--encoder_dim", type=list, nargs="+", default=[])
    parser.add_argument("--embed_dim", type=int, default=0)

    # model parameters
    parser.add_argument("--con_temperature", type=float, default=0.5)
    parser.add_argument("--dist_temperature", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.07)
    parser.add_argument("--start_rectify_epoch", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--n_views", type=int, default=2, help="number of views")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")

    # training setting
    parser.add_argument("--batch_size", type=int, default=256, help="batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20, help="epochs to warmup learning rate")
    parser.add_argument("--data_norm",type=str,default="standard",choices=["standard", "min-max", "l2-norm"])
    parser.add_argument("--train_time", type=int, default=1)

    # optimizer parameters
    parser.add_argument("--weight_decay", type=float,default=0,help="Initial value of the weight decay. (default: 0)")

    parser.add_argument("--lr",type=float, default=None,metavar="LR",help="learning rate (absolute lr)")
    # data loader and logger
    parser.add_argument("--dataset",type=str,default="LandUse21",choices=["LandUse21","Scene15",],)
    parser.add_argument("--data_path", type=str, default="../../Datasets/Multi_View/", help="path to your folder of dataset")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--output_dir",type=str,default="./",help="path where to save, empty for no saving",)

    parser.add_argument("--print_freq", default=10)

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--pin_mem",action="store_true",help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--fp_ratio", type=float, default=0.0)
    parser.add_argument("--noise_ratio", type=float, default=0.5)
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--divide", action="store_true")
    parser.add_argument("--candy", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--singular_thresh", type=float, default=0.2)
    return parser


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)

    # dataset
    dataset = load_dataset(args)
    sampler_train = IncompleteDatasetSampler(dataset, seed=args.seed)
    if args.batch_size > len(sampler_train):
        args.batch_size = len(sampler_train)
    data_loader_train = torch.utils.data.DataLoader(dataset,sampler=sampler_train,batch_size=args.batch_size,pin_memory=args.pin_mem,drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=False,pin_memory=args.pin_mem,drop_last=False)

    # models
    if args.divide:
        model = DivideModel(n_views=args.n_views,layer_dims=args.encoder_dim,n_classes=args.n_classes,drop_rate=args.drop_rate, args=args)
    elif args.candy:
        model = CandyModel(n_views=args.n_views,layer_dims=args.encoder_dim,n_classes=args.n_classes,drop_rate=args.drop_rate, args=args)
    else:
        model = NoisyModel(n_views=args.n_views,layer_dims=args.encoder_dim,n_classes=args.n_classes,drop_rate=args.drop_rate, args=args)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    state_logger.write(
        "\n>> Start training {}-th initial, seed: {},".format(args.train_id, args.seed)
    )

    train_state_dict = dict()

    best_acc = 0.
    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        training_log = train_one_epoch(model,data_loader_train,optimizer,device,epoch,args)
        eval_result = evaluate(epoch, model, data_loader_test, device, args)

        if args.print_this_epoch:
            train_state_dict[epoch] = eval_result

        if args.print_this_epoch:
            state_logger.write(
                "Epoch {} K-means: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}".format(
                    epoch,
                    eval_result["acc"],
                    eval_result["nmi"],
                    eval_result["ari"],
                )
            )

        # # csv
        # with open(result_path, 'a+') as csvfile:
        #     writer = csv.writer(csvfile)
        #     if not (os.path.exists(result_path) and os.path.getsize(result_path) > 0):
        #         writer.writerow(list(training_log.keys()) + list(eval_result.keys()))
        #     writer.writerow(list(training_log.values()) + list(eval_result.values()))

        if eval_result["acc"] > best_acc:
            best_acc = eval_result["acc"]
            best_result = eval_result
    print(best_result)

    # best_file = os.path.join(args.output_dir, 'results_con.csv')
    # with open(best_file, 'a+') as csvfile:
    #     writer = csv.writer(csvfile)
    #     if not (os.path.exists(best_file) and os.path.getsize(best_file) > 0):
    #         writer.writerow(['con_temperature', 'dist_temperature'] + list(best_result.keys()))
    #     writer.writerow([args.con_temperature, args.dist_temperature] + list(best_result.values()))
    return train_state_dict


def main(args):
    start_time = time.time()

    result_avr_def = {"nmi": [], "ari": [], "f": [], "acc": []}

    result_dict = dict()

    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale
        args.min_lr = args.min_blr * batch_scale
    exp_name = args.output_dir + time.strftime("%Y-%m-%d_%H-%M-%S")
    state_logger = utils.FileLogger("log_train.txt")

    for t in range(args.train_time):
        args.train_id = t
        train_state_dict = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        for epoch, train_state in train_state_dict.items():
            result_dict.setdefault(epoch, copy.deepcopy(result_avr_def))
            for k, v in train_state.items():
                result_dict[epoch][k].append(v)

    for epoch, res in result_dict.items():
        for k, v in res.items():
            x = np.asarray(v) * 100

            res[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    for epoch, result_avr in result_dict.items():
        print("Epoch {}:".format(epoch))
        kmeans_r = result_avr
        state_logger.write("title,acc,ari,nmi")
        state_logger.write(
            "concat,{:.2f}({:.2f}),{:.2f}({:.2f}),{:.2f}({:.2f})".format(
                *kmeans_r["acc"], *kmeans_r["ari"], *kmeans_r["nmi"]
            )
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, "FullLoader"):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    args.embed_dim = args.encoder_dim[0][-1]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = os.path.join(args.output_dir, f"result_{time_str}.csv")
    main(args)
