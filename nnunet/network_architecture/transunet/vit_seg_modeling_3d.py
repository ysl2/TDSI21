# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _triple
from scipy import ndimage
from nnunet.network_architecture.transunet import vit_seg_configs as configs
from nnunet.network_architecture.transunet.vit_seg_modeling_resnet_skip_3d import ResNetV2_3D


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu,
          "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(
            config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(
            config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(
            config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings3D(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings3D, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = np.array(img_size)

        if config.get("patch_size") is not None:   # ResNet
            resnet_ft_size = img_size // 16
            patch_size = np.array(config.patch_size)
            embedding_kernel_size = np.maximum(resnet_ft_size // patch_size, 1)
            self.embedding_shape = resnet_ft_size // embedding_kernel_size
            n_patches = np.prod(resnet_ft_size // embedding_kernel_size)
            self.hybrid = True
        else:
            raise ValueError("Must define config.patch_size for TransUNet3D")

        if self.hybrid:
            self.hybrid_model = ResNetV2_3D(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
                cin=config.cin)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv3d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=embedding_kernel_size,
            stride=embedding_kernel_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()

            query_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class AttentionEncoder(nn.Module):
    def __init__(self, config, vis):
        super(AttentionEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer3D(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer3D, self).__init__()
        self.embeddings = Embeddings3D(config, img_size=img_size)
        self.encoder = AttentionEncoder(config, vis)
        self.embedding_shape = list(self.embeddings.embedding_shape.astype(int))

    def forward(self, input_ids, reshape_output=False):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(
            embedding_output)  # (B, n_patch, hidden)

        # reshape to (B, hidden, d, h, w) if needed
        if reshape_output:
            encoded = encoded.permute(0, 2, 1) # (B, hidden, n_patches)
            b, hidden, _ = encoded.shape
            # (B, hidden, d, h, w)
            encoded = encoded.reshape(b, hidden, *self.embedding_shape).contiguous()
        return encoded, attn_weights, features


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Decoder3DBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead3D(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv3d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(
            scale_factor=upsampling, mode="trilinear") if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            # re-select the skip channels according to n_skip
            for i in range(4-self.config.n_skip):
                skip_channels[3-i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            Decoder3DBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, encoded_ft, features=None, do_ds=False):
        x = self.conv_more(encoded_ft)
        all_levels_x = []
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            if do_ds:
                all_levels_x.append(x)
        if do_ds:
            return all_levels_x
        else:
            return x


class VisionTransformer3D(nn.Module):
    def __init__(self, config, img_size=224, num_classes=6, zero_head=False, vis=False, do_ds=False):
        super(VisionTransformer3D, self).__init__()
        self.img_size = config.img_size if config.get(
            "img_size") is not None else img_size
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer3D(config, self.img_size, vis)
        self.decoder = DecoderCup3D(config)
        self.segmentation_heads = nn.ModuleList([
            SegmentationHead3D(
                in_channels=config['decoder_channels'][i],
                out_channels=config['n_classes'],
                kernel_size=3)
            for i in range(len(config['decoder_channels']))
        ])
        self.config = config
        # Support deep supervision
        self._deep_supervision = True
        # Performe deep supervision ?
        self.do_ds = do_ds

    def forward(self, x):
        x, attn_weights, features = self.transformer(x, reshape_output=True)
        x = self.decoder(x, features, do_ds=self.do_ds)
        if self.do_ds:
            all_logits = []
            for i, ft_map in enumerate(x):
                all_logits.append(self.segmentation_heads[i](ft_map))
            return all_logits
        else:
            logits = self.segmentation_heads[-1](x)
            return logits


if __name__ == "__main__":
    config = configs.get_r50_b16_3d_config(6)
    config.img_size = (96, 192, 192)
    model = VisionTransformer3D(config, do_ds=True).cuda()
    x = torch.rand(2, 1, 96, 192, 192).cuda()
    y = model(x)
    print(model)
