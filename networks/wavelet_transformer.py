"""
Wavelet Transformer for Image Super-Resolution and Denoising

Jin Zhu (jin.zhu@cl.cam.ac.uk) @ 2021.08
"""
from networks.wavelet_common import PytorchDWT, PytorchDWTInverse
import math
import torch
import torch.nn.functional as F
from torch import nn
import copy


class WaveletTransformerBasic(nn.Module):
    """
    Image-in-image-out wavelet transformer (data_mode = 'image')
    Input and output shape: N  x C x H x W image batches

    Wavelet tokens in/out (data_mode = 'coeffs'):
    Input and output shape: N x P x C x h x w wavelet coeffs

    All wavelet tokens are processed by a simple ViT.

    Head:
        Get wavelet tokens:
        N x C x H x W -> N x P x C x h x w, where P = (H//h)*(W//w)
        Reshape the input wavelet tokens.
        N x P x C x h x w -> N x P x T
    Body:
        ViT N x P x T -> N x P x T
    Tail:
        N x P x T -> N x P x C x h x w
        Reconstruct image:
        N x P x C x h x w -> N x C x H x W

    """
    def __init__(self, paras):
        super(WaveletTransformerBasic, self).__init__()
        # basic settings
        self.input_channel = paras.input_channel        # C
        self.wavelet_level = paras.wavelet_level
        self.wavelet_patch_size = paras.wavelet_hr_patch_size
        self.num_tokens = int(4 ** self.wavelet_level)
        self.patch_dim = int(self.wavelet_patch_size / (2 ** self.wavelet_level))
        self.token_dim = int(self.input_channel * self.patch_dim * self.patch_dim)

        # ViT parameters
        self.n_heads = paras.wtb_num_heads
        self.n_layers = paras.wtb_num_layers
        self.hidden_dim_factor = paras.wtb_hidden_dim_factor
        self.dropout_rate = paras.wtb_dropout_rate
        self.no_mlp = paras.wtb_no_mlp
        self.no_norm = paras.wtb_no_norm
        self.no_pos = paras.wtb_no_pos
        self.pos_every = paras.wtb_pos_every

        # wavelet settings
        self.data_mode = paras.wt_data_mode
        self.residual_scale = paras.residual_scale
        # head: wavelet tokens
        self.wavelet_kernel = paras.wavelet_kernel
        self.head = PytorchDWT(self.wavelet_level, self.wavelet_kernel)

        # ViT
        self.body = VisionTransformer(
            input_dim=self.token_dim,
            output_dim=self.token_dim,
            num_tokens=self.num_tokens,
            embedding_dim=self.token_dim,
            num_heads=self.n_heads,
            num_layers=self.n_layers,
            hidden_dim=self.token_dim * self.hidden_dim_factor,
            dropout_rate=self.dropout_rate,
            no_norm=self.no_norm,
            no_mlp=self.no_mlp,
            pos_every=self.pos_every,
            no_pos=self.no_pos
        )

        # tail: reconstruct image from wavelet tokens
        self.tail = PytorchDWTInverse(self.wavelet_kernel)

    def forward(self, x, s):
        res = x
        if self.data_mode in ['image']:
            # x: N x C x H x W -> N x P x C x h x w
            x = self.head(x)
        # x: N x P x C x h x w -> N x P x T
        x = x.view(-1, self.num_tokens, self.token_dim).contiguous()
        x = self.body(x, s)
        # x: N x P x T -> N x P x C x h x w
        x = x.view(-1, self.num_tokens, self.input_channel, self.patch_dim, self.patch_dim)

        if self.data_mode in ['image']:
            # x: -> N x C x H x W
            x = self.tail(x)
            if self.residual_scale > 0.:
                x = x + res * self.residual_scale
        return x


class WaveletTransformerRiver(nn.Module):
    """
    Image-in-image-out wavelet transformer (data_mode = 'image')
    Input and output shape: N  x C x H x W image batches

    Wavelet tokens in/out (data_mode = 'coeffs'):
    Input and output shape: N x P x C x h x w wavelet coeffs

    ViTs are like rivers: steam ViT process 4 tokens, then 16 tokens, then 64 ...
    # This can be done in the config_file
        1. mlp is for scale embedding, so mlp for the first level ViTs only?
        2. what about position embedding? -> should always be with pos_embedding
    Head:
        Get wavelet tokens:
        N x C x H x W -> N x P x C x h x w, where P = (H//h)*(W//w)
        Reshape the input wavelet tokens.
        N x P x C x h x w -> N x P x T
    Body:
        A module list of ViTs
        Levels of ViTs and for each level: if wavelet_level = 4
            1: N x 4 x T    P/4 ViTs
            2: N x 16 x T   P/16 ViTs
            3: N x 64 x T   P/64 ViTs
            4: N x 256 x T  P/256 ViTs
        ViT N x P x T -> N x P x T
    Tail:
        N x P x T -> N x P x C x h x w
        Reconstruct image:
        N x P x C x h x w -> N x C x H x W

    """
    def __init__(self, paras):
        super(WaveletTransformerRiver, self).__init__()
        # basic settings
        self.input_channel = paras.input_channel  # C
        self.wavelet_level = paras.wavelet_level
        self.wavelet_patch_size = paras.wavelet_hr_patch_size
        self.num_tokens = int(4 ** self.wavelet_level)
        self.patch_dim = int(self.wavelet_patch_size / (2 ** self.wavelet_level))
        self.token_dim = int(self.input_channel * self.patch_dim * self.patch_dim)

        # ViT parameters for each level
        self.n_heads = self.__align_parameter__(paras.wtr_num_heads)
        self.n_layers = self.__align_parameter__(paras.wtr_num_layers)
        self.hidden_dim_factor = self.__align_parameter__(paras.wtr_hidden_dim_factor)
        self.dropout_rate = self.__align_parameter__(paras.wtr_dropout_rate)
        self.no_mlp = self.__align_parameter__(paras.wtr_no_mlp)
        self.no_norm = self.__align_parameter__(paras.wtr_no_norm)
        self.no_pos = self.__align_parameter__(paras.wtr_no_pos)
        self.pos_every = self.__align_parameter__(paras.wtr_pos_every)

        # wavelet settings
        self.data_mode = paras.wt_data_mode
        self.residual_scale = paras.residual_scale
        # head: wavelet tokens
        self.wavelet_kernel = paras.wavelet_kernel
        self.head = PytorchDWT(self.wavelet_level, self.wavelet_kernel)

        # body, ViTs in each level
        self.body = nn.ModuleList()
        for l in range(self.wavelet_level):
            input_dim = self.token_dim
            output_dim = self.token_dim
            num_tokens = 4 ** (l + 1)
            embedding_dim = self.token_dim
            self.body.append(
                VisionTransformer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_tokens=num_tokens,
                    embedding_dim=embedding_dim,
                    num_heads=self.n_heads[l],
                    num_layers=self.n_layers[l],
                    hidden_dim=embedding_dim * self.hidden_dim_factor[l],
                    dropout_rate=self.dropout_rate[l],
                    no_norm=self.no_norm[l],
                    no_mlp=self.no_mlp[l],
                    pos_every=self.pos_every[l],
                    no_pos=self.no_pos[l]
                ))

        # tail: reconstruct image from wavelet tokens
        self.tail = PytorchDWTInverse(self.wavelet_kernel)

    def forward(self, x, s):
        res = x
        if self.data_mode in ['image']:
            # x: N x C x H x W -> N x P x C x h x w
            x = self.head(x)
        # x: N x P x C x h x w -> N x P x T
        x = x.view(-1, self.num_tokens, self.token_dim).contiguous()

        for l in range(self.wavelet_level):
            # Pi
            num_tokens = 4 ** (l + 1)
            # N x P x T -> Ni x Pi x T; Ni = N * P/Pi
            x = x.view(-1, num_tokens, self.token_dim).contiguous()
            s_l = s.repeat(int(self.num_tokens/num_tokens), 1)
            # feed to vit
            x = self.body[l](x, s_l)
            # x: Ni x Pi x T -> N x P x T
            x = x.view(-1, self.num_tokens, self.token_dim).contiguous()

        # x: N x P x T -> N x P x C x h x w
        x = x.view(-1, self.num_tokens, self.input_channel, self.patch_dim, self.patch_dim)

        if self.data_mode in ['image']:
            # x: -> N x C x H x W
            x = self.tail(x)
            if self.residual_scale > 0.:
                x = x + res * self.residual_scale
        return x

    def __align_parameter__(self, a):
        if not isinstance(a, (list, tuple)):
            return [a] * self.wavelet_level
        else:
            return a


class WaveletTransformerPyramid(nn.Module):
    """
    Input and output shape:
        mode='image': N x C x H x W.
        mode='coeffs': N x P x C x h x w

    Notice that for H, W should be <= 64 because of GPU memory limitation

    ViTs are like a Pyramid:
        1. each ViT process 4 tokens and generate a bigger one;
        2. the next level do the same on bigger tokens
        ...
        3. finally get the image

    H, W > 64 lead to heavy GPU usage.
        Probably we can crop the tokens in patches, this works for DWT/iDWT. unlike fft

    Head:
        If 'image':
            Get wavelet tokens:
            N x C x H x W -> N x P x C x h x w, where P = (H//h)*(W//w)
        Reshape the input wavelet tokens.
        N x P x C x h x w -> N x P x T
    Body:
        A module list of ViTs
        Levels of ViTs and for each level: if wavelet_level = 4
            1: N x 4 x T    P/4 ViTs
            2: N x 4 x T   P/16 ViTs
            3: N x 4 x T   P/64 ViTs
            4: N x 4 x T  P/256 ViTs
        ViT N x P x T -> N x P x T
    Tail:
        N x P x T -> N x P x C x h x w
        if 'image':
            Reconstruct image:
            N x P x C x h x w -> N x C x H x W

    """
    def __init__(self, paras):
        super(WaveletTransformerPyramid, self).__init__()
        # basic settings
        self.input_channel = paras.input_channel  # C
        self.wavelet_level = paras.wavelet_level
        self.wavelet_patch_size = paras.wavelet_hr_patch_size
        self.num_tokens = int(4 ** self.wavelet_level)
        self.patch_dim = int(self.wavelet_patch_size / (2 ** self.wavelet_level))
        self.token_dim = int(self.input_channel * self.patch_dim * self.patch_dim)

        # ViT parameters for each level
        self.n_heads = self.__align_parameter__(paras.wtp_num_heads)
        self.n_layers = self.__align_parameter__(paras.wtp_num_layers)
        self.hidden_dim_factor = self.__align_parameter__(paras.wtp_hidden_dim_factor)
        self.dropout_rate = self.__align_parameter__(paras.wtp_dropout_rate)
        self.no_mlp = self.__align_parameter__(paras.wtp_no_mlp)
        self.no_norm = self.__align_parameter__(paras.wtp_no_norm)
        self.no_pos = self.__align_parameter__(paras.wtp_no_pos)
        self.pos_every = self.__align_parameter__(paras.wtp_pos_every)

        # wavelet settings
        self.data_mode = paras.wt_data_mode
        self.residual_scale = paras.residual_scale
        # head: wavelet tokens
        self.wavelet_kernel = paras.wavelet_kernel
        self.head = PytorchDWT(self.wavelet_level, self.wavelet_kernel)

        # body, ViTs in each level
        self.body = nn.ModuleList()
        for l in range(self.wavelet_level):
            input_dim = self.token_dim * (4 ** l)
            output_dim = self.token_dim * (4 ** l)
            num_tokens = 4
            embedding_dim = self.token_dim * (4 ** l)
            self.body.append(
                VisionTransformer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_tokens=num_tokens,
                    embedding_dim=embedding_dim,
                    num_heads=self.n_heads[l],
                    num_layers=self.n_layers[l],
                    hidden_dim=embedding_dim * self.hidden_dim_factor[l],
                    dropout_rate=self.dropout_rate[l],
                    no_norm=self.no_norm[l],
                    no_mlp=self.no_mlp[l],
                    pos_every=self.pos_every[l],
                    no_pos=self.no_pos[l]
                ))

        # tail: reconstruct image from wavelet tokens
        # also be used for each level
        self.idwt = PytorchDWTInverse(self.wavelet_kernel)

    def forward(self, x, s):
        res = x
        if self.data_mode in ['image']:
            # x: N x C x H x W -> N x P x C x h x w
            x = self.head(x)
        N = x.size(0)

        for l in range(self.wavelet_level):
            # x: N x Pi x C x hi x wi -> N x Pi x Ti
            x = x.view(x.size(0), x.size(1), -1).contiguous()
            # split the tokens to windows with 4 tokens
            num_coeffs_groups = int(x.size(1) / 4)
            # x: N x Pi x Ti -> Nl x 4 x Ti; Nl = N * Pi/4
            x = x.view(-1, 4, x.size(-1)).contiguous()
            s_l = s.repeat(num_coeffs_groups, 1)
            # feed tokens to vit
            x = self.body[l](x, s_l)
            # Nl x 4 x Ti -> Nl x 4 x C x hi x wi
            wl = hl = self.patch_dim * (2 ** l)
            x = x.view(-1, 4, self.input_channel, hl, wl).contiguous()
            # idwt: -> Nl x C x 2*hl x 2* wl
            x = self.idwt(x)
            # x: Nl x C x w*hl x 2*wl -> N x P x C x h x w
            x = x.view(N, -1, self.input_channel, hl*2, wl*2).contiguous()

        # the image has been reconstructed as N x 1 x C x H x W
        x = x[:, 0]
        if self.residual_scale > 0.:
            x = x + res * self.residual_scale
        return x

    def __align_parameter__(self, a):
        if not isinstance(a, (list, tuple)):
            return [a] * self.wavelet_level
        else:
            return a


class WaveletTransformerStairs(nn.Module):
    """
    Input and output shape:
        mode='image': N x C x H x W.
        mode='coeffs': [N x 1 x C x hn x wn,
                        N x 3 x C x hn x wn,
                        ...,
                        N x 3 x C x h1 x w1]

    Notice that for H, W should be <= 64 because of GPU memory limitation

    ViTs are like a stair:
        1. on each level there is only one ViT which processes 4 tokens;
            N x 4 x Ti -> N x 1 x T(i+1);
        2. next level three new wavelet tokens will be added;
        ...
        3. finally get the image

    H, W > 64 lead to heavy GPU usage.
        Probably we can crop the tokens in patches, this works for DWT/iDWT. unlike fft

    Head:
        If 'image':
            Get wavelet tokens:
            N x C x H x W -> [N x 1 x C x hn x wn,
                              N x 3 x C x hn x wn,
                              ...,
                              N x 3 x C x h1 x w1]
    Body:
        Get N x 4 x C x hi x wi tokens, reshape to N x 4 x Ti
        A module list of ViTs
        Levels of ViTs and for each level: if wavelet_level = 4
            1: N x 4 x T    1 ViT
            2: N x 4 x T    1 ViT
            3: N x 4 x T    1 ViT
            4: N x 4 x T    1 ViT
        ViT N x 4 x Ti -> N x 4 x Ti
        Reshape: N x 4 x Ti -> N x 4 x C x hi x wi
        idwt: N x 4 x C x hi x wi -> N x 1 x C x h(i+1) x w(i+1)
    Tail:
        N x P x T -> N x P x C x h x w
        if 'image':
            Reconstruct image:
            N x P x C x h x w -> N x C x H x W

    """

    def __init__(self, paras):
        super(WaveletTransformerStairs, self).__init__()
        # basic settings
        self.input_channel = paras.input_channel  # C
        self.wavelet_level = paras.wavelet_level
        self.wavelet_patch_size = paras.wavelet_hr_patch_size
        self.num_tokens = int(4 ** self.wavelet_level)
        self.patch_dim = int(self.wavelet_patch_size / (2 ** self.wavelet_level))
        self.token_dim = int(self.input_channel * self.patch_dim * self.patch_dim)

        # ViT parameters for each level
        self.n_heads = self.__align_parameter__(paras.wts_num_heads)
        self.n_layers = self.__align_parameter__(paras.wts_num_layers)
        self.hidden_dim_factor = self.__align_parameter__(paras.wts_hidden_dim_factor)
        self.dropout_rate = self.__align_parameter__(paras.wts_dropout_rate)
        self.no_mlp = self.__align_parameter__(paras.wts_no_mlp)
        self.no_norm = self.__align_parameter__(paras.wts_no_norm)
        self.no_pos = self.__align_parameter__(paras.wts_no_pos)
        self.pos_every = self.__align_parameter__(paras.wts_pos_every)

        # wavelet settings
        self.data_mode = paras.wt_data_mode
        self.residual_scale = paras.residual_scale
        # head: wavelet tokens
        self.wavelet_kernel = paras.wavelet_kernel
        self.head = PytorchDWT(self.wavelet_level, self.wavelet_kernel, 'part')

        # body, ViTs in each level
        self.body = nn.ModuleList()
        for l in range(self.wavelet_level):
            input_dim = self.token_dim * (4 ** l)
            output_dim = self.token_dim * (4 ** l)
            num_tokens = 4
            embedding_dim = self.token_dim * (4 ** l)
            self.body.append(
                VisionTransformer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_tokens=num_tokens,
                    embedding_dim=embedding_dim,
                    num_heads=self.n_heads[l],
                    num_layers=self.n_layers[l],
                    hidden_dim=embedding_dim * self.hidden_dim_factor[l],
                    dropout_rate=self.dropout_rate[l],
                    no_norm=self.no_norm[l],
                    no_mlp=self.no_mlp[l],
                    pos_every=self.pos_every[l],
                    no_pos=self.no_pos[l]
                ))

        # tail: reconstruct image from wavelet tokens
        # also be used for each level
        self.idwt = PytorchDWTInverse(self.wavelet_kernel)

    def forward(self, x, s):
        res = x
        if self.data_mode in ['image']:
            # x: N x C x H x W -> N x P x C x h x w
            x = self.head(x)

        ca = x[0]   # N x 1 x C x h0 x w0
        for l in range(self.wavelet_level):
            # N x 4 x C x hi x wi
            coeffs = torch.cat([ca, x[l+1]], dim=1)
            # coeffs: N x 4 x C x hi x wi -> N x 4 x Ti
            coeffs = coeffs.view(coeffs.size(0), coeffs.size(1), -1).contiguous()

            coeffs = self.body[l](coeffs, s)
            # coeffs: N x 4 x Ti -> N x 4 x C x hi x wi
            wl = hl = self.patch_dim * (2 ** l)
            coeffs = coeffs.view(-1, 4, self.input_channel, hl, wl)
            # ca: N x C x 2*hi x 2*wi
            ca = self.idwt(coeffs)
            # -> N x 1 x C x h x w
            ca = ca.unsqueeze(1)

        # the image has been reconstructed as N x 1 x C x H x W
        x = ca[:, 0]
        if self.residual_scale > 0.:
            x = x + res * self.residual_scale
        return x

    def __align_parameter__(self, a):
        if not isinstance(a, (list, tuple)):
            return [a] * self.wavelet_level
        else:
            return a


class VisionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens, embedding_dim, num_heads, num_layers, hidden_dim,
                 dropout_rate=0, no_norm=False, no_mlp=False, pos_every=False, no_pos=False):
        """
        Tokens generated from images are with shape: N x P x C x h x w, where is N is the batch size.
        The original image shape is N x C x H x W.
        Before feeding to the transformer, the tokens are reshaped as N x P x T.
        :param input_dim: The dim of each token, T = C x h x w
        :param output_dim: The dim of each output token, should be the same as input_dim
        :param num_tokens: The number of tokens P = (H / h) * (W / w)
        :param embedding_dim: dim after linear embedding, should be the same as input_dim
        :param num_heads: For multi-head self attention
        :param num_layers: number of encoder and seg_loss_mode layers
        :param hidden_dim: features in hidden layers
        :param dropout_rate:
        :param no_norm: No layer norm
        :param no_mlp: No MLP layer
        :param pos_every: apply position embedding at each step
        :param no_pos: No position embedding
        """
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert input_dim == output_dim
        self.no_norm = no_norm
        self.no_mlp = no_mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.pos_every = pos_every
        self.seq_length = num_tokens

        self.input_dim = input_dim
        self.out_dim = output_dim

        self.no_pos = no_pos

        if not self.no_mlp:
            self.linear_encoding = nn.Linear(self.input_dim, embedding_dim)
            self.mlp_tail = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            # N x 1 tensor float -> N x E embedding
            self.sr_scale_embed = nn.Linear(1, embedding_dim * self.seq_length)
            # self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, sr_scale):
        """

        :param x: N x P x T tensor float
        :param sr_scale: N x 1 tensor float
        :return: N x P x T tensor
        """

        # N x C x H x W -> P x N x T: T = C x h x w, P = H*W / h*w
        # x: N x P x T -> P x N x T; sr_scale: N x 1
        x = x.transpose(0, 1).contiguous()
        if not self.no_mlp:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            # The query_embed is replaced by a linear layer, which takes the sr scale as input
            # sr_scale: N x 1, query_embed: P x N x T (same as x)
            query_embed = self.sr_scale_embed(sr_scale).view(-1, self.seq_length, self.embedding_dim).transpose(0, 1).contiguous()
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)

        if not self.no_mlp:
            x = self.mlp_tail(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.input_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        # the position_ids will be part of the state_dict, so it will be saved
        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")