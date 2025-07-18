import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


from ..bricks import ConvModule, TransformerBlock
from ..builder import PROJECTIONS
import sys
sys.path.append(r"/tad_work/Pointnet_Pointnet2_pytorch/models")
import pointnet2_cls_msg, pointnet_cls


@PROJECTIONS.register_module()
class Conv1DTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        attn_cfg=None,  # n_head n_mha_win_size attn_pdrop
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class PC3DFeatExtractor(nn.Module):  #self-defined high dimensional convolution feature extractor. however, it cannot be used due to high computation demand.
    """
    Extract per-frame spatial features from 5D+T input (B, H, W, D, C, T) and
    project to (B, embed_dim, T) for temporal processing.
    """
    def __init__(
        self,
        spatial_in_channels=3,
        spatial_conv_channels=(32, 64, 128),
        embed_dim=256
    ):
        super().__init__()
        # Build 3D spatial extractor: multiple Conv3d + ReLU
        layers = []
        in_ch = spatial_in_channels
        for out_ch in spatial_conv_channels:
            layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        # Global pooling to (1,1,1)
        layers.append(nn.AdaptiveAvgPool3d(1))
        self.spatial_extractor = nn.Sequential(*layers)
        # Linear projector from last 3D conv channels to embed_dim
        self.frame_projector = nn.Linear(spatial_conv_channels[-1], embed_dim)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, H, W, D, C, T)
        returns: Tensor of shape (B, embed_dim, T)
        """
        B, H, W, D, C, T = x.shape
        # Move T into batch and reorder to (B*T, C, H, W, D)
        x = x.permute(0, 5, 4, 1, 2, 3).reshape(B * T, C, H, W, D)
        # Extract spatial features and pool
        feat = self.spatial_extractor(x)          # (B*T, C_last, 1,1,1)
        feat = feat.view(B * T, -1)               # (B*T, C_last)
        # Project to embed_dim
        feat = self.frame_projector(feat)         # (B*T, embed_dim)
        # Restore to (B, embed_dim, T)
        feat = feat.view(B, T, -1).permute(0, 2, 1)
        return feat



@PROJECTIONS.register_module()
class PCTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        prep_cfg=None,
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        attn_cfg=None,  # n_head n_mha_win_size attn_pdrop
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        # point cloud feature extractor
        # self.PCFE = PC3DFeatExtractor(spatial_in_channels=prep_cfg["input_channels"], spatial_conv_channels=prep_cfg["conv_channels"], embed_dim=prep_cfg["embed_channels"])
        # self.PCFE = pointnet2_cls_msg.get_model(num_class=6, normal_channel=False) # pointnet++ feature extractor, unusable due to high computational demand too.
        self.PCFE = pointnet_cls.get_model(k=2, normal_channel=False)

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature extraction
        B, N, C, T = x.shape
        # x = x.permute(0, 3, 2, 1)    # → [B, T, C, N]
        # x = x.reshape(B * T, C, N)   # → [B*T, C, N]
        # x = self.PCFE(x)
        # x = x.view(B, T, 1024).permute(0, 2, 1)  # → [B, 1024, T]

        # x = x.permute(0, 2, 1, 3)  # -> [B, C, N, T]
        # outputs = []
        # with torch.no_grad():
        #     for t in range(T):
        #         _,_,out = self.PCFE(x[:, :, :, t])  # x: [B, C, N, T]
        #         outputs.append(out)               # 每帧单独处理
        #         torch.cuda.empty_cache()
        # x = torch.stack(outputs, dim=-1)      # → [B, 1024, T]

        x = x.permute(0, 3, 2, 1)
        #print(x.shape)
        x = x.reshape(B * T, C, N)  # → (B*T, 4, 406)
        _, _, x, _ = self.PCFE(x)  #here output of PCFE is of 256 dims
        x = x.view(B, T, 256)    # → (B, T, 256), 256 is the output dim of PCFE

    


        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
