import random
import torch

from .import functional as F


# Based on https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/transforms.py
class GroupNormalize(object):
    """
    Normalize a list of images.
    Args:
        mean: mean of image for each channel
        std: std of image for each channel
        multi_clips: set True if input is a list of videos instead of one video
    """
    def __init__(self, mean, std, multi_clips=False):
        self.mean = mean
        self.std = std
        self.multi_clips = multi_clips

    def __call__(self, tensor):
        if self.multi_clips:
            tensors = [self.work(t) for t in tensor]
            return torch.stack(tensors, dim=0)
        else:
            return self.work(tensor)

    def work(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make it more efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor


class SpatialRandomCrop(object):
    """Crops a random spatial crop in a spatio-temporal
    numpy or tensor input [Channel, Time, Height, Width]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size

    def __call__(self, tensor):
        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)
        cropped = tensor[:, :, y1:y1 + h, x1:x1 + h]
        return cropped
