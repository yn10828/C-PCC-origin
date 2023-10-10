import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from models.pointops.functions import pointops
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from einops import rearrange, repeat
from models.utils import index_points
import math
from models.icosahedron2sphere import icosahedron2sphere
chamfer_dist = chamfer_3DDist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from copy import deepcopy
def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    k = min(k, pairwise_distance.size()[1])

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx,k

def get_graph_feature(data,k):
    xyz = data

    # x = x.squeeze()
    idx,k = knn(xyz, k=k)  # (batch_size, num_points, k)

    torch.cuda.empty_cache()
    batch_size, num_points, _ = idx.size()
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = xyz.size()

    xyz = xyz.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)

    # k1=xyz.size()[2]
    # gxyz
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)
    # xyz
    xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #lxyz_norm
    neighbor_lxyz_norm = torch.norm(neighbor_gxyz - xyz, dim=3, keepdim=True)

    feature = torch.cat((xyz, neighbor_gxyz, neighbor_lxyz_norm), dim=3)

    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature

#------------------white-box Transformer-----------------#

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()

        self.args = args
        self.heads = args.heads
        inner_dim = args.dim_head * self.heads
        self.scale = args.dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(args.dropout)

        project_out = not (self.heads == 1 and inner_dim == in_dim)

        # self.w = nn.Linear(in_dim, inner_dim, bias=False)

        self.w = nn.Conv1d(in_dim, inner_dim, 1)
        # self.qkv = nn.Conv1d(in_dim, inner_dim, 1)

        # self.to_out = nn.Conv1d(inner_dim, out_dim, 1)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, out_dim, 1)
        ) if project_out else nn.Identity()
        # self.to_out = nn.Conv1d(inner_dim, out_dim, 1)

    def forward(self, feats, sample_idx):

        # (b, n, c)
        f = feats
        f = index_points(feats, sample_idx)
        # f = self.w(f)
        w = self.w(f).permute(0, 2, 1).contiguous()

        # (h d) = c  ---'b n (h d) -> b h n d'
        w = rearrange(w, 'b n (h d) -> b h n d', h=self.heads)
        # attention
        dots = torch.matmul(w, w.transpose(-1, -2))

        attn = self.attend(dots * self.scale)
        attn = self.dropout(attn)
        # (b, h, m, d)
        out = torch.matmul(attn, w)

        #(b, n, c)
        fea = rearrange(out, 'b h n d  -> b n (h d) ')


        feature = fea.permute(0, 2, 1).contiguous()
        feature = self.to_out(feature)
        # torch.cuda.empty_cache()
        output = feature + f

        return output

class FeedForward(nn.Module):
    def __init__(self, in_dim,  args):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, in_dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = args.step_size
        self.lambd = args.lambd

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.GroupNorm(args.ngroups, args.dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.dim, args.dim, 1)
        )

    def forward(self, x):

         # x shape: (b, c, m)
        # x = index_points(x, sample_idx)
        b, c, m = x.size()

        # Reshape x to (b * m, c) for matrix multiplication
        x_reshaped = x.view(b * m, c)

        # Compute D^T * D * x
        x1 = torch.einsum('ij,bj->bi', [self.weight, x_reshaped])
        grad_1 = torch.einsum('ij,bi->bj', [self.weight, x1])

        # Compute D^T * x
        grad_2 = torch.einsum('ij,bi->bj', [self.weight, x_reshaped])

        # Compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        # Reshape grad_update back to (b, c, m)
        grad_update = grad_update.reshape(b, c, m)

        out = F.relu(x + grad_update)

        out = self.to_latent(out)

        out = self.mlp_head(out)



        return out



class GeometricLayer(nn.Module):
    def __init__(self,args):
        super(GeometricLayer, self).__init__()
        # 确定输入的点云信息
        # neighboursnum = 16
        # self.neighboursnum = neighboursnum
        self.in_dim = args.dim

        self.pre_nn = nn.Sequential(
            nn.Conv2d(4, args.dim, 1)
        )

        self.localConv = nn.Sequential(
            nn.Conv2d(args.dim*2+1, args.hidden_dim,1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )
        self.qkv = nn.Conv1d(args.dim, args.dim, 1)
        self.pre = nn.Sequential(
            nn.Conv2d(3, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )
        self.attn_nn = nn.Sequential(
            nn.Conv2d(args.dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.hidden_dim, args.dim, 1)
        )
        self.semConv =  nn.Sequential(
            nn.Conv1d(args.dim, args.hidden_dim,1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.dim, 1)
        )
        # self.fullConv = nn.Sequential(
        #     nn.Conv1d(args.hidden_dim*2, args.hidden_dim, 1),
        #     nn.GroupNorm(args.ngroups, args.hidden_dim),
        #     nn.ReLU(),
        #     nn.Conv1d(args.hidden_dim, args.dim, 1)
        # )


        # self.fullConv = nn.Conv1d(args.dim * 2, args.dim, 1)

    def forward(self, q_xyzs, k_xyzs, sam_feats,  xyz_feats, knn_idx,mask):
                    # sampled_xyzs, xyzs, sampled_feats, feats, knn_idx, mask
        # q_xyzs: (b, 3, m), k_xyzs: (b, 3, n), knn_idx and mask: (b, m, k)

        # (b, 3, m, k)
        knn_xyzs = index_points(k_xyzs, knn_idx)
        # (b, 3, m, k) k=16
        k = knn_xyzs.shape[-1]
        repeated_xyzs = q_xyzs[..., None].repeat(1, 1, 1, k)

        # (b, 3, m, k)
        direction = F.normalize(knn_xyzs - repeated_xyzs, p=2, dim=1)
        # (b, 1, m, k)
        distance = torch.norm(knn_xyzs - repeated_xyzs, p=2, dim=1, keepdim=True)
        # (b, 4, m, k)
        local_pattern = torch.cat((direction, distance), dim=1)

        # (b, c, m, k)   L1:(1,c=8,2564,16)
        position_embedding = self.pre_nn(local_pattern)

        # (b, c, m)
        position_embedding = position_embedding.sum(dim=-1)

        # def forward( self, f_in ):    # f_in:(B, C, m)
        neighbor_f_in = get_graph_feature(position_embedding, k)   # (B, C, m, k)
        Intra_channal = self.localConv( neighbor_f_in )
        Intra_channal = Intra_channal.max(dim=-1, keepdim=False)[0]


        #Gro_embedding
        #(b, c, m)
        query = self.qkv(sam_feats)

        # (b, c, n)
        # key = self.qkv(xyz_feats)
        # value = self.qkv(xyz_feats)

        # (b, c, m, k)
        key = index_points(self.qkv(xyz_feats), knn_idx)
        value = index_points(self.qkv(xyz_feats), knn_idx)
        #  (b, m, n)
        # scores = torch.einsum('bdm,bdn->bmn', q, key) / self.in_dim**.5
        # (b, c, m, k)
        pos_enc = self.pre(q_xyzs.unsqueeze(-1) - knn_xyzs)
        # attention
        attn = self.attn_nn(query.unsqueeze(-1) - key + pos_enc)

        # # (b, c, m, k)
        # attn = self.attn_nn(position_embedding)
        #  (b, m, n)
        attn = attn / math.sqrt(key.shape[1])
        # attn = attn[..., None].repeat(1, 1, 1, k)
        # mask
        mask_value = -(torch.finfo(attn.dtype).max)
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)

        # scores = scores.squeeze(-1)
        # scscoresores = torch.softmax(scores, dim=-1)
        # fgt = torch.einsum('bmn,bdn->bdm', scores, value) #(1,8,m)
        fgt = torch.einsum('bcmk, bcmk->bcm', attn, value + pos_enc)

        Inter_channal = self.semConv(fgt) #(1,64,m)

        # feature = torch.cat( (Intra_channal, Inter_channal), dim= 1 )

        # feature = self.fullConv(feature)

        # return feature   #, Inter_channal
        return Intra_channal , Inter_channal


class DownsampleLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(DownsampleLayer, self).__init__()

        self.args = args
        self.k = args.k
        self.downsample_rate = args.downsample_rate[layer_idx]

        self.pre_conv = nn.Conv1d(args.dim, args.dim, 1)

        self.AttentionLayer = AttentionLayer(args.dim, args.dim, args)
        self.FeedForward = FeedForward(args.dim,  args)
        self.Geo_embedding_nn = GeometricLayer(args)

        self.post_conv = nn.Conv1d(args.dim * 3 , args.dim, 1)


    def get_density(self, sampled_xyzs, xyzs):
        # input: (b, 3, m), (b, 3, n)

        batch_size = xyzs.shape[0]
        sample_num = sampled_xyzs.shape[2]
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()
        # (b, sample_num, 3)
        sampled_xyzs_trans = sampled_xyzs.permute(0, 2, 1).contiguous()

        # find the nearest neighbor in sampled_xyzs_trans: (b, n, 1)--该函数的作用是查找 xyzs_trans 中每个点在 sampled_xyzs_trans 中的最近邻。返回的结果 ori2sample_idx 是一个形状为 (b, n, 1) 的张量，表示每个点在 sampled_xyzs_trans 中的最近邻索引。
        ori2sample_idx = pointops.knnquery_heap(1, sampled_xyzs_trans, xyzs_trans)

        # (b, sample_num)
        downsample_num = torch.zeros((batch_size, sample_num)).to(device)
        for i in range(batch_size):
            uniques, counts = torch.unique(ori2sample_idx[i], return_counts=True)
            downsample_num[i][uniques.long()] = counts.float()

        # (b, m, k)  # knn_idx将每个点的邻居点按照距离排序，选取最近的K个邻居点
        knn_idx = pointops.knnquery_heap(self.k, xyzs_trans, sampled_xyzs_trans).long()
        # --------------------------------huan cun-------------
        torch.cuda.empty_cache()

        # mask: (m)
        expect_center = torch.arange(0, sample_num).to(device)
        # (b, m, k)
        expect_center = repeat(expect_center, 'm -> b m k', b=batch_size, k=self.k)
        # (b, 1, m, k)
        real_center = index_points(ori2sample_idx.permute(0, 2, 1).contiguous(), knn_idx)
        # (b, m, k)
        real_center = real_center.squeeze(1)
        # mask those points that not belong to collapsed points set--最终得到的 mask 张量表示哪些点属于下采样后的采样点云中的中心点，哪些点不属于.
        mask = torch.eq(expect_center, real_center)

        # (b, 3, m, k)
        knn_xyzs = index_points(xyzs, knn_idx)
        # (b, 1, m, k)----表示每个点与其最近邻点之间的欧几里得距离
        distance = torch.norm(knn_xyzs - sampled_xyzs[..., None], p=2, dim=1, keepdim=True)
        # mask
        mask_value = 0
        distance.masked_fill_(~mask[:, None], mask_value)
        # (b, m)
        distance = distance.sum(dim=-1).squeeze(1)
        # (b, m)---表示每个点与其最近邻点之间的平均距离。
        mean_distance = distance / downsample_num

        return downsample_num, mean_distance, mask, knn_idx


    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n), features: (b, cin, n) L1:xyzs(1,3,1815), features(1,8,1815)
        if self.k > xyzs.shape[2]:
            self.k = xyzs.shape[2]

        sample_num = round(xyzs.shape[2] * self.downsample_rate)   #L1: sample_num:605
        # (b, n, 3)
        xyzs_trans = xyzs.permute(0, 2, 1).contiguous()

        # FPS, (b, sample_num)
        sample_idx = pointops.furthestsampling(xyzs_trans, sample_num).long()
        # (b, 3, sample_num)
        sample_idx.to(device)
        sampled_xyzs = index_points(xyzs, sample_idx)

        # get density
        downsample_num, mean_distance, mask, knn_idx = self.get_density(sampled_xyzs, xyzs)

        identity = feats
        feats = self.pre_conv(feats)
        # (b, c, sample_num)
        sampled_feats = index_points(feats, sample_idx)

        # embedding  L1:(1,8,m)
        Attention_embedding = self.AttentionLayer(feats, sample_idx)
        creat_embedding = self.FeedForward(Attention_embedding)
        Intra_channal , Inter_channal = self.Geo_embedding_nn(sampled_xyzs, xyzs, sampled_feats, feats, knn_idx, mask)
        agg_embedding = torch.cat((creat_embedding, Intra_channal , Inter_channal ), dim=1)
        # (b, c, m)
        agg_embedding = self.post_conv(agg_embedding)
        # agg_embedding = self.post_conv(creat_embedding)
        # agg_embedding = self.post_conv(density_embedding)
        # residual connection: (b, c, m)--对聚合后的特征进行残差连接的操作
        sampled_feats = agg_embedding + index_points(identity, sample_idx)

        return sampled_xyzs, sampled_feats, downsample_num, mean_distance



class EdgeConv(nn.Module):
    def __init__(self, args, in_fdim, out_fdim):
        super(EdgeConv, self).__init__()

        self.k = args.k

        self.conv = nn.Sequential(
            nn.Conv2d(2*in_fdim, args.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(args.hidden_dim, out_fdim, 1)
        )


    def knn(self, feats):
        inner = -2 * torch.matmul(feats.transpose(2, 1), feats)
        xx = torch.sum(feats ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        # (b, n, k)
        knn_idx = pairwise_distance.topk(k=self.k, dim=-1)[1]

        return knn_idx


    def get_graph_features(self, feats):
        dim = feats.shape[1]
        if dim == 3:
            # (b, n, 3)
            feats_trans = feats.permute(0, 2, 1).contiguous()
            # (b, n, k)
            knn_idx = pointops.knnquery_heap(self.k, feats_trans, feats_trans).long()
        else:
            # it needs a huge memory cost
            knn_idx = self.knn(feats)

        #     --------------huan  cun
        torch.cuda.empty_cache()
        # (b, c, n, k)
        knn_feats = index_points(feats, knn_idx)
        repeated_feats = repeat(feats, 'b c n -> b c n k', k=self.k)
        # (b, 2c, n, k)
        graph_feats = torch.cat((knn_feats-repeated_feats, repeated_feats), dim=1)

        return graph_feats


    def forward(self, feats):
        # input: (b, c, n)
        if feats.shape[2] < self.k:
            self.k = feats.shape[2]

        graph_feats = self.get_graph_features(feats)
        # (b, cout*g, n, k)
        expanded_feats = self.conv(graph_feats)
        # (b, cout*g, n)
        expanded_feats = torch.max(expanded_feats, dim=-1)[0]

        return expanded_feats




class SubPointConv(nn.Module):
    def __init__(self, args, in_fdim, out_fdim, group_num):
        super(SubPointConv, self).__init__()

        assert args.sub_point_conv_mode in ['mlp', 'edge_conv']
        self.mode = args.sub_point_conv_mode
        self.hidden_dim = args.hidden_dim
        self.group_num = group_num
        self.group_in_fdim = in_fdim // group_num
        self.group_out_fdim = out_fdim // group_num

        # mlp
        if self.mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv2d(self.group_in_fdim, self.hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim, self.group_out_fdim, 1)
            )
        else:
            # edge_conv
            self.edge_conv = EdgeConv(args, in_fdim, out_fdim)


    def forward(self, feats):
        if self.mode == 'mlp':
            # per-group conv: (b, cin, n, g)
            feats = rearrange(feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()
            # (b, cout, n, g)
            expanded_feats = self.mlp(feats)
        else:
            # (b, cout*g, n)
            expanded_feats = self.edge_conv(feats)
            # shuffle: (b, cout, n, g)
            expanded_feats = rearrange(expanded_feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()

        return expanded_feats





class XyzsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None):
        super(XyzsUpsampleLayer, self).__init__()

        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]
        else:
            self.upsample_rate = upsample_rate

        # each point has fixed 43 candidate directions, size (43, 3)  ???
        hypothesis, _ = icosahedron2sphere(1)
        hypothesis = np.append(np.zeros((1,3)), hypothesis, axis=0)
        self.hypothesis = torch.from_numpy(hypothesis).float().to(device)


        # weights
        self.weight_nn = SubPointConv(args, args.dim, 43*self.upsample_rate, self.upsample_rate)
        # self.weight_nn = nn.ConvTranspose1d(args.dim, 43*self.upsample_rate, 1, 1, bias=False)


        # scales
        # self.scale_nn = SubPointConv(args, args.dim, 1*self.upsample_rate, self.upsample_rate)
        self.scale_nn = nn.ConvTranspose1d(args.dim, 1*self.upsample_rate, 1, 1, bias=False)


    def forward(self, xyzs, feats):
        # xyzs: (b, 3, n)  feats (b, c, n)
        batch_size = xyzs.shape[0]
        points_num = xyzs.shape[2]


        # (b, 43*u, n)
        pre_weights = self.weight_nn(feats)
        # (b, 43, 1, n, u)
        weights = pre_weights.unsqueeze(2)
        weights = F.softmax(weights, dim=1)

        # (b, 43, 3, n, u)
        hypothesis = repeat(self.hypothesis, 'h c -> b h c n u', b=batch_size, n=points_num, u=self.upsample_rate)
        weighted_hypothesis = weights * hypothesis
        # (b, 3, n, u)
        directions = torch.sum(weighted_hypothesis, dim=1)
        # normalize
        directions = F.normalize(directions, p=2, dim=1)

        # (b, 1*u, n)
        pre_scales = self.scale_nn(feats)
        # (b, 1, n, u)
        scales = rearrange(pre_scales, 'b (c u) n -> b c n u', u=self.upsample_rate).contiguous()

        # (b, 3, n, u)
        # deltas = directions * pre_scales
        deltas = directions * scales

        # (b, 3, n, u)
        repeated_xyzs = repeat(xyzs, 'b c n -> b c n u', u=self.upsample_rate)
        upsampled_xyzs = repeated_xyzs + deltas

        return upsampled_xyzs


class FeatsUpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx, upsample_rate=None, decompress_normal=False):
        super(FeatsUpsampleLayer, self).__init__()


        if upsample_rate == None:
            self.upsample_rate = args.max_upsample_num[layer_idx]   #8,8,8
        else:
            self.upsample_rate = upsample_rate

        self.group_num = self.upsample_rate


        # weather decompress normal
        self.decompress_normal = decompress_normal
        if self.decompress_normal:
            self.out_fdim = 3
        else:
            self.out_fdim = args.dim

        self.group_out_fdim = self.out_fdim * self.upsample_rate

        # self.feats_nn = SubPointConv(args, args.dim, self.out_fdim * self.upsample_rate, self.upsample_rate)

        self.feats_nn = nn.ConvTranspose1d(args.dim , self.group_out_fdim ,1 ,1 , bias=False)   # point-wise splitting


    def forward(self, feats):
        # (b, c, n, u) =(1, 8, n, 8)
        # upsampled_feats = self.feats_nn(feats)


        # feats_nn1 lose n
        # (1, 64,  )
        # n = self.group_out_fdim
        upsampled_feats = self.feats_nn(feats)
        upsampled_feats = rearrange(upsampled_feats, 'b (c u) n -> b c n u', u = self.upsample_rate).contiguous()


        if self.decompress_normal == False:
            # shortcut
            repeated_feats = repeat(feats, 'b c n -> b c n u', u=self.upsample_rate)
            # (b, c, n, u)
            upsampled_feats = upsampled_feats + repeated_feats

        return upsampled_feats




class UpsampleLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(UpsampleLayer, self).__init__()

        self.xyzs_upsample_nn = XyzsUpsampleLayer(args, layer_idx)
        self.feats_upsample_nn = FeatsUpsampleLayer(args, layer_idx)


    def forward(self, xyzs, feats):
        upsampled_xyzs = self.xyzs_upsample_nn(xyzs, feats)
        upsampled_feats = self.feats_upsample_nn(feats)

        return upsampled_xyzs, upsampled_feats