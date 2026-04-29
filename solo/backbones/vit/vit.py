# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from timm https://github.com/rwightman/pytorch-image-models/blob/master/timm/

from timm.models.registry import register_model
import timm

@register_model
def vit_tiny(patch_size=16, **kwargs):
    """ViT-Small is the smallest available DINOv3 ViT variant"""
    model = timm.create_model(
        'vit_small_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0,
    )
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model = timm.create_model(
        'vit_small_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0,
    )
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model = timm.create_model(
        'vit_base_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0,
    )
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model = timm.create_model(
        'vit_large_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0,
    )
    return model