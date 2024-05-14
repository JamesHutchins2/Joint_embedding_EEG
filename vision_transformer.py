# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from masks.utils import apply_masks


def get_2d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Generates sine-cosine positional embeddings for 1D data.

    Args:
    - embed_dim (int): The dimensionality of the embedding for each position.
    - length (int): The length of the sequence for which to generate embeddings.
    - cls_token (bool): Whether to include an additional position for the class token.

    Returns:
    - numpy.ndarray: A matrix of size [length (+1 if cls_token), embed_dim] containing the positional embeddings.
    """
    # Create a 1D grid representing the positions in the sequence
    grid_l = np.arange(length, dtype=np.float32).reshape([1, length])

    # Generate the positional embeddings from the grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_l)

    # If a class token is used, prepend a zero embedding
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
    print("pos_embed shape: ", pos_embed.shape)

    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, pos):
    #print("position embedding grid embed dim: ", embed_dim)
    """
    Generates sine-cosine positional embeddings given a grid of positions.

    Args:
    - embed_dim (int): The dimensionality of the embedding for each position.
    - pos (numpy.ndarray): A 1D array of positions to be encoded.

    Returns:
    - numpy.ndarray: A matrix of size [len(pos), embed_dim] containing the positional embeddings.
    """
    # Ensure the embedding dimension is even
    assert embed_dim % 2 == 0

    # Generate the scales for the sine and cosine functions
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)  # Scaling factors for each dimension

    pos = pos.reshape(-1)  # Flatten the position array if not already flat

    # Calculate the dot product of positions and omega, for sine and cosine separately
    out = np.einsum('m,d->md', pos, omega)  # Outer product to get (M, D/2)

    # Generate sine and cosine embeddings
    emb_sin = np.sin(out)  # Sine part of the embedding
    emb_cos = np.cos(out)  # Cosine part of the embedding

    # Concatenate sine and cosine embeddings to form the final embedding
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # Final embedding (M, D)

    return emb



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class PatchEmbed1D(nn.Module):
    """
    A class for converting 1D data into patch embeddings.
    
    It divides the data into patches and projects each patch into an embedding space.
    
    Parameters:
    - time_len: The length of the time series data.
    - patch_size: The size of each patch.
    - in_chans: Number of input channels (features per time point).
    - embed_dim: The dimensionality of the output embedding space.
    """
    
    def __init__(self,img_size=512, time_len=512, patch_size=4, in_chans=128, embed_dim=256):
        time_len=512 
        patch_size=4 
        in_chans=128 
        embed_dim=1024
        # Initialize the parent class (nn.Module)
        super().__init__()
        
        # Calculate the number of patches by dividing the total length by the patch size
        num_patches = time_len // patch_size
        
        print("num_patches: ", num_patches)
        
        # Initialize attributes
        self.patch_size = patch_size
        self.time_len = time_len
        self.num_patches = num_patches

        # Define a convolutional layer to project the input data into the embedding space
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        print("size of projection layer: ", self.proj)


    def forward(self, x):
        """
        Forward pass of the module.
        
        Parameters:
        - x: The input data of shape (Batch size, Channels, Length of the data)
        
        Returns:
        - The patch embeddings of the input data.
        """
        # Ensure x is in the correct shape: (Batch size, Channels, Data length)
        B, C, V = x.shape
        
        # Project the input data into the embedding space and reshape
        # The 'proj' layer outputs (Batch size, embed_dim, Num patches)
        # We transpose to make it (Batch size, Num patches, embed_dim) for further processing
        x = self.proj(x).transpose(1, 2).contiguous()
        
        return x

class PatchEmbed(nn.Module):
    """ Convert 1D signal data to Patch Embedding """
    def __init__(self, img_size=512, patch_size=4, in_chans=128, embed_dim=1024):
        
        #print("image size input: ", img_size)
        #print("patch size input: ", patch_size)
        #print("in_chans input: ", in_chans)
        
        in_chans = 512
        img_size = 128
        
        super().__init__()
        self.img_size = img_size    # Represents the length of the signal
        self.patch_size = patch_size
        self.in_chans = in_chans    # Number of channels (formerly treated as input channels)
        self.embed_dim = embed_dim
        self.num_patches = img_size // patch_size

        # Using 1D Convolution here to treat each segment of data as a patch
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Print expected input shape on initialization
        print(f"Initialized PatchEmbed with expected input shape: [Batch_size, {in_chans}, {img_size}]")

    def forward(self, x):
        B, C, L = x.shape  # Check input dimensions
        if C != self.in_chans or L != self.img_size:
            raise ValueError(f"Input shape should be [batch_size, {self.in_chans}, {self.img_size}], but got [{B}, {C}, {L}]")
        x = self.proj(x).transpose(1, 2)  # Transpose to get [Batch_size, Num_patches, Embed_dim]
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=512, in_chans=128, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=1024,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #print("number of patches in predictor: ", num_patches)
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, 32, predictor_embed_dim),
                                                requires_grad=False)
        print("predictor pos embed shape: ", self.predictor_pos_embed.shape)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      32,
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        #print("VIT PREDICTOR FORWARD")
        
        if not isinstance(masks_x, list):
            #print("mask 1")
            masks_x = [masks_x]

        if not isinstance(masks, list):
            #print("mask 2")
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        #print("predictor embedding called")
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        #print("predictor positional embedding called")
        print("x shape before positional embedding: ", x.shape)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape
        #print("concatinating mask tokens")
        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        #print("mask tokens repeated, calling pred tokens")
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        #print("calling predictor blocks")
        for blk in self.predictor_blocks:
            x = blk(x)
        #print("calling predictor norm")
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        #print("predict  tokens")
        x = x[:, N_ctxt:]
        #print("predictor projection")
        x = self.predictor_proj(x)
        #print("predictor projection done")
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[512],
        patch_size=16,
        in_chans=128,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        img_size=[512]
        patch_size=4
        in_chans=128
        embed_dim=1024
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        
        #get patch_embed input size
        
        
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, 1024), requires_grad=False)
        #print("pos_embed shape: ",self.pos_embed.shape)
        #print("self.pos_embed.shape[-1]: ",self.pos_embed.shape[-1])
        #print("int(self.patch_embed.num_patches**.5): ", int(self.patch_embed.num_patches**.5))
        pos_embed = get_2d_sincos_pos_embed(1024,
                                            32,
                                            cls_token=False)
        print("pe shape: ", pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        #print("VIT CREATED")

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        #print("forward call x shape: ", x.shape)
        #print("VIT FORWARD CALLED")
        x = x.transpose(1, 2)
        if masks is not None:
            print("masks called")
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        
        #print("shape of x after patch embedding: ", x.shape)
        B, N, D = x.shape
        #print("mask appiled")
        # -- add positional embedding to x
        print("x shape before positional embedding: ", x.shape)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        
        #print("positional embedding interpolated, shape is: ", pos_embed.shape)
        #print("x shape: ", x.shape)
        x = x + pos_embed
        #print("positional embedding added")
        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        #print("transformer has been reached")
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        print("positional embedding shape: ", pos_embed.shape)
        #print("x shape: ", x.shape)
        npatch = x.shape[1] - 1
        print("interpolate npatches: ", npatch)
        N = pos_embed.shape[1] - 1
        print("interpolate N: ", N)
        
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        print("class emb shape: ", class_emb.shape)
        pos_embed = pos_embed[:, 1:]
        print("pos_embed shape: ", pos_embed.shape)
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}