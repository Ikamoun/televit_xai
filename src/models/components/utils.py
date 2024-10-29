"""
Utility functions for segmentation models.
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_margins_to_image(img, margin_size):
    margin_left = img.crop((0, 0, margin_size, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_right = img.crop((img.width - margin_size, 0, img.width, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_top = img.crop((0, 0, img.width, margin_size)).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom = img.crop((0, img.height - margin_size, img.width, img.height)).transpose(Image.FLIP_TOP_BOTTOM)

    margin_top_left = img.crop((0, 0, margin_size, margin_size)).transpose(Image.FLIP_LEFT_RIGHT).transpose(
        Image.FLIP_TOP_BOTTOM)
    margin_top_right = img.crop((img.width - margin_size, 0, img.width, margin_size)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_left = img.crop((0, img.height - margin_size, margin_size, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_right = img.crop(
        (img.width - margin_size, img.height - margin_size, img.width, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    concat_img = Image.new('RGB', (img.width + margin_size * 2, img.height + margin_size * 2))

    concat_img.paste(img, (margin_size, margin_size))
    concat_img.paste(margin_left, (0, margin_size))
    concat_img.paste(margin_right, (img.width + margin_size, margin_size))
    concat_img.paste(margin_top, (margin_size, 0))
    concat_img.paste(margin_bottom, (margin_size, img.height + margin_size))
    concat_img.paste(margin_top_left, (0, 0))
    concat_img.paste(margin_top_right, (img.width + margin_size, 0))
    concat_img.paste(margin_bottom_left, (0, img.height + margin_size))
    concat_img.paste(margin_bottom_right, (img.width + margin_size, img.height + margin_size))

    return concat_img


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales is not None:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        if len(self.scales) == 0:
            return logits

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max
