import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device_ids = [0, 1, 2, 3]
import torch
import torch.nn as nn

import sys

sys.path.append(os.getcwd())
from models.autoencoder import AutoEncoder
import time
import argparse
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.utils import AverageMeter, str2bool
from dataset.dataset import CompressDataset
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from torch.optim.lr_scheduler import StepLR
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    start = time.time()

    if args.batch_size > 1:
        print('The performance will degrade if batch_size is larger than 1!')

    if args.compress_normal == True:
        args.in_fdim = 6

    # load data
    train_dataset = CompressDataset(data_path=args.train_data_path, cube_size=args.train_cube_size,
                                    batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)

    val_dataset = CompressDataset(data_path=args.val_data_path, cube_size=args.val_cube_size,
                                  batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size)

    # set up folders for checkpoints
    str_time = datetime.now().isoformat()
    print('Experiment Time:', str_time)
    checkpoint_dir = os.path.join(args.output_path, str_time, 'ckpt')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create the model
    model = AutoEncoder(args)

    if args.use_wandb:
        wandb.watch(model)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # model.cuda()
    # model = nn.parallel.DataParallel(model)
    model = model.to(device)

    # print('Training Arguments:', args)
    # print('Model Architecture:', model)

    # optimizer for autoencoder
    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=args.lr)
    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    # optimizer for entropy bottleneck  para of entropy
    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)

    # best validation metric/
    best_val_chamfer_loss = float('inf')

    # train
    for epoch in range(args.epochs):
        epoch_loss = AverageMeter()
        epoch_chamfer_loss = AverageMeter()
        epoch_density_loss = AverageMeter()
        epoch_pts_num_loss = AverageMeter()
        epoch_latent_xyzs_loss = AverageMeter()
        epoch_normal_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()

        model.train()

        for i, input_dict in enumerate(train_loader):
            # input: (b, n, c) 其中 b 表示批次大小，n 表示点的数量，c 表示每个点的特征维度。b:1,n:3,c:36548
            input = input_dict['xyzs'].to(device)
            # input: (b, c, n)
            input = input.permute(0, 2, 1).contiguous()

            # compress normal
            if args.compress_normal == True:
                normals = input_dict['normals'].to(device).permute(0, 2, 1).contiguous()
                input = torch.cat((input, normals), dim=1)

            # model forward
            # with autocast():
            decompressed_xyzs, loss, loss_items, bpp = model(input)

            epoch_loss.update(loss.item())
            epoch_chamfer_loss.update(loss_items['chamfer_loss'])
            epoch_density_loss.update(loss_items['density_loss'])
            epoch_pts_num_loss.update(loss_items['pts_num_loss'])
            epoch_latent_xyzs_loss.update(loss_items['latent_xyzs_loss'])
            epoch_normal_loss.update(loss_items['normal_loss'])
            epoch_bpp_loss.update(loss_items['bpp_loss'])

            # # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute gradient and do SGD step
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            #
            # scaler.step(optimizer)
            # scaler.update()

            # update the parameters of entropy bottleneck
            aux_loss = model.feats_eblock.loss()
            if args.quantize_latent_xyzs == True:
                aux_loss += model.xyzs_eblock.loss()
            epoch_aux_loss.update(aux_loss.item())

            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()

            if args.use_wandb:
                wandb.log({"train loss": epoch_loss.get_avg(),
                           "avg chamfer loss": epoch_chamfer_loss.get_avg(), \
                           "avg density loss": epoch_density_loss.get_avg(), \
                           "avg pts num loss": epoch_pts_num_loss.get_avg(), \
                           "avg latent xyzs loss": epoch_latent_xyzs_loss.get_avg(), \
                           "avg normal loss": epoch_normal_loss.get_avg(), \
                           "avg bpp loss": epoch_bpp_loss.get_avg(), \
                           "avg aux loss": epoch_aux_loss.get_avg()})
            # print loss
            if (i + 1) % args.print_freq == 0:
                print("train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                      "avg density loss: %f, avg pts num loss: %f, avg latent xyzs loss: %f, "
                      "avg normal loss: %f, avg bpp loss: %f, avg aux loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss.get_avg(),
                       epoch_chamfer_loss.get_avg(),
                       epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(), epoch_latent_xyzs_loss.get_avg(),
                       epoch_normal_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))

        scheduler_steplr.step()

        # print loss
        interval = time.time() - start
        print("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
              "avg density loss: %f, avg pts num loss: %f, avg latent xyzs loss: %f, "
              "avg normal loss: %f, avg bpp loss: %f, avg aux loss: %f" %
              (epoch + 1, args.epochs, interval / 60, interval % 60, epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
               epoch_density_loss.get_avg(), epoch_pts_num_loss.get_avg(), epoch_latent_xyzs_loss.get_avg(),
               epoch_normal_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))

        # if args.use_wandb:
        #     wandb.log({"train loss": epoch_loss.get_avg(),
        #                "avg chamfer loss": epoch_chamfer_loss.get_avg(), \
        #                "avg density loss": epoch_density_loss.get_avg(), \
        #                "avg pts num loss": epoch_pts_num_loss.get_avg(), \
        #                "avg latent xyzs loss": epoch_latent_xyzs_loss.get_avg(), \
        #                "avg normal loss": epoch_normal_loss.get_avg(), \
        #                "avg bpp loss": epoch_bpp_loss.get_avg(), \
        #                "avg aux loss": epoch_aux_loss.get_avg()})

        # validation
        model.eval()
        val_chamfer_loss = AverageMeter()
        val_normal_loss = AverageMeter()
        val_bpp = AverageMeter()
        with torch.no_grad():
            for input_dict in val_loader:
                # xyzs: (b, n, c)
                input = input_dict['xyzs'].to(device)
                # (b, c, n)
                input = input.permute(0, 2, 1).contiguous()

                # compress normal
                if args.compress_normal == True:
                    normals = input_dict['normals'].to(device).permute(0, 2, 1).contiguous()
                    input = torch.cat((input, normals), dim=1)
                    args.in_fdim = 6

                # gt_xyzs
                gt_xyzs = input[:, :3, :].contiguous()

                # model forward
                decompressed_xyzs, loss, loss_items, bpp = model(input)
                # calculate val loss and bpp
                d1, d2, _, _ = chamfer_dist(gt_xyzs.permute(0, 2, 1).contiguous(),
                                            decompressed_xyzs.permute(0, 2, 1).contiguous())
                chamfer_loss = d1.mean() + d2.mean()
                val_chamfer_loss.update(chamfer_loss.item())
                val_normal_loss.update(loss_items['normal_loss'])
                val_bpp.update(bpp.item())

        # print loss
        print("val epoch: %d/%d, val bpp: %f, val chamfer loss: %f, val normal loss: %f" %
              (epoch + 1, args.epochs, val_bpp.get_avg(), val_chamfer_loss.get_avg(), val_normal_loss.get_avg()))

        # save checkpoint
        cur_val_chamfer_loss = val_chamfer_loss.get_avg()
        if cur_val_chamfer_loss < best_val_chamfer_loss or (epoch + 1) % args.save_freq == 0:
            model_name = 'ckpt-best.pth' if cur_val_chamfer_loss < best_val_chamfer_loss else 'ckpt-epoch-%02d.pth' % (
                        epoch + 1)
            model_path = os.path.join(checkpoint_dir, model_name)
            torch.save(model.state_dict(), model_path)
            # update best val chamfer loss
            if cur_val_chamfer_loss < best_val_chamfer_loss:
                best_val_chamfer_loss = cur_val_chamfer_loss


# 这段代码的目的是将train_args中的属性复制到model_args中，以确保model_args对象具有与train_args相同的属性和值。
def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='semantickitti', type=str, help='shapenet or semantickitti')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='the performance will degrade if batch_size is larger than 1!')
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float,
                        help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int,
                        help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    # normal compression
    parser.add_argument('--compress_normal', default=False, type=str2bool, help='whether compress normals')
    # compress latent xyzs
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str,
                        help='latent xyzs conv mode, mlp or edge_conv')
    # sub_point_conv mode
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')
    parser.add_argument('--output_path', default='./output/result_retrain', type=str, help='output path')
    # wandb
    parser.add_argument('--wandb_dir', type=str, default='./wandb')
    parser.add_argument('--use_wandb', default='use_wandb', action='store_true')
    # parser.add_argument('--runname', type=str, default='sem825_01')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['shapenet', 'semantickitti']

    if train_args.dataset == 'shapenet':
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(train_args, model_args)

    print(model_args.use_wandb)
    if model_args.use_wandb:
        import wandb

        wandb.init(config=model_args, project='C-PCC-Ablation-train', name='CRATE+3-3-3', dir=model_args.wandb_dir)

    train(model_args)
