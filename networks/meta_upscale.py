import torch
import math
from torch import nn


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


class MetaUpSampler(nn.Module):

    def __init__(self, n_feats, n_colors, kernel_size):
        super(MetaUpSampler, self).__init__()
        self.P2W = Pos2Weight(inC=n_feats, outC=n_colors)
        self.outC = n_colors
        self.inC = n_feats
        self.kernel_size = kernel_size

    def forward(self, lr_features, sr_scale):
        N, C, inH, inW = lr_features.shape
        device = lr_features.device
        # ## predict local weights
        pos_mat, mask = input_matrix_wpn_new(inH, inW, sr_scale)
        mask = mask.to(device)
        pos_mat = pos_mat.to(device)

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))

        up_x = self.repeat_x(lr_features, sr_scale)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = nn.functional.unfold(up_x, self.kernel_size, padding=1)
        scale_int = math.ceil(sr_scale)
        local_weight = self.repeat_weight(local_weight, scale_int, inH, inW)

        cols = cols.contiguous().view(
            cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2), 1
        ).permute(0, 1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(
            inH, scale_int, inW, scale_int, -1, self.outC
        ).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(
            scale_int ** 2, inH * inW, -1, self.outC
        )

        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        out = out.contiguous().view(
            N, scale_int, scale_int, self.outC, inH, inW
        ).permute(0, 3, 4, 1, 5, 2)
        out = out.contiguous().view(
            N, self.outC, scale_int * inH, scale_int * inW
        )

        # ## mask
        out = torch.masked_select(out, mask)
        out = out.contiguous().view(
            N, self.outC, int(sr_scale * inH), int(sr_scale * inW)
        )
        return out

    @staticmethod
    def repeat_x(x, scale):
        scale_int = math.ceil(scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    @staticmethod
    def repeat_weight(weight, scale, inh, inw):
        k = int(math.sqrt(weight.size(0)))
        outw = inw * scale
        outh = inh * scale
        weight = weight.view(k, k, -1)
        scale_w = (outw + k - 1) // k
        scale_h = (outh + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)

        weight = weight[0:outh, 0:outw, :]

        return weight


def input_matrix_wpn_new(inH, inW, scale, add_scale=True):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW = int(scale * inH), int(scale * inW)
    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)

    ####projection  coordinate  and caculate the offset
    h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)

    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)

    i = 1
    h, w, _ = pos_mat.size()
    while (pos_mat[i][0][0] >= 1e-6 and i < h):
        i = i + 1

    j = 1
    # pdb.set_trace()
    h, w, _ = pos_mat.size()
    while (pos_mat[0][j][1] >= 1e-6 and j < w):
        j = j + 1

    pos_mat_small = pos_mat[0:i, 0:j, :]

    pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
    if add_scale:
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)  ###(inH*inW*scale_int**2, 4)
        pos_mat_small = torch.cat((scale_mat.view(1, -1, 1), pos_mat_small), 2)

    return pos_mat_small, mask_mat  ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
