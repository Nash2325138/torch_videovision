import numbers
import random

import cv2
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy
import torch
import torchvision

from . import functional as F


# Based on https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/transforms.py
class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=['center', 'corner'], manually_set_random=False):
        """
        Args:
            input_size:
                Image will be resized into input_size after cropped
            scales (list of float):
                The valid ratios of the input image's shorter side to determine side length of cropping boxes
            manually_set_random:
                Set crop size and random size only when first __call__ or manually calling set_random()
                before the transform operation. Used when there are multiple modalities where the composed transform
                need to be created each time in __getitem__() and applied to different modalities.
            fix_crop:
                If True, then only positions defined in more_fix_crop will be sampled,
                If False, the start (offsets) of the crop is randomly sampled.
            more_fix_crop:
                A list defining what kind of crops is going to be sampled if `fix_crop` is
                True, allowing options are `center`, `corner`, `edge`, and `quarter`
        """
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

        self.manually_set_random = manually_set_random
        if manually_set_random:
            self.crop_w, self.crop_h, self.offset_w, self.offset_h = None, None, None, None

    def set_random(self, im_size):
        # Randomly set crop size and offset
        self.crop_w, self.crop_h, self.offset_w, self.offset_h = self._sample_crop_size(im_size)

    def __call__(self, img_group):

        im_size = img_group[0].size

        # Random for each called
        if not self.manually_set_random:
            crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        # Random only when first called or manually set
        else:
            # First called
            if self.crop_w is None:
                self.set_random(im_size)
            crop_w, crop_h, offset_w, offset_h = self.crop_w, self.crop_h, self.offset_w, self.offset_h

        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def get_n_crop_candidate(more_fix_crop):
        count_map = {'corner': 4, 'center': 1, 'edge': 4, 'quarter': 4}
        ans = 0
        for kind in more_fix_crop:
            ans += count_map(kind)
        return ans

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """
        Extract candidates of offset (start of x and y) for cropping

        Args:
            more_fix_crop: (list of str)
                Defines what kind of cropping offsets to apply.
                It could have 'corner', 'center', 'edge', and 'quarter' in this list
        """
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        if 'corner' in more_fix_crop:
            ret.append((0, 0))  # upper left
            ret.append((4 * w_step, 0))  # upper right
            ret.append((0, 4 * h_step))  # lower left
            ret.append((4 * w_step, 4 * h_step))  # lower right

        if 'center' in more_fix_crop:
            ret.append((2 * w_step, 2 * h_step))  # center

        if 'edge' in more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

        if 'quarter' in more_fix_crop:
            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


# Modified from https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py
class GroupOverSample(object):
    """
    Spatially multi-crops a video (including flipped cropped)
    Args:
        crop_size: `int`, `(int, int)`, or '[int, int]'
            The size to crop
        more_fix_crop: list of str
            List of cropping types, please refer to GroupMultiScaleCrop.fill_fix_offset for valid options

    Methods:
        __call__: return a list of cropped videos (a video is of list of images)
    """
    def __init__(self, crop_size, more_fix_crop=['center', 'quarter', 'edge'], is_flow=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        self.more_fix_crop = more_fix_crop
        self.is_flow = is_flow

    def set_flow(self, is_flow=True):
        self.is_flow = is_flow

    def __call__(self, img_group):
        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        oversample_groups = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if self.is_flow and i % 2 == 0:  # Handle flow data
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_groups.append(normal_group)
            oversample_groups.append(flip_group)
        return oversample_groups


class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """ Horizontally flip the list of given images randomly with a probability 0.5

    Args:
        manually_set_random: set if random flip only when initialized or manually calling set_random()
            before the transform operation. Used when there are multiple modalities where the composed transform
            need to be created each time in __getitem__() and applied to different modalities.
    """
    def __init__(self, is_flow=False, manually_set_random=False):
        self.is_flow = is_flow
        self.manually_set_random = manually_set_random
        if manually_set_random:
            self.flip = self.set_random()

    def set_flow(self, is_flow=True):
        self.is_flow = is_flow

    def set_random(self):
        self.flip = random.random() < 0.5

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if (not self.manually_set_random and random.random() < 0.5) or (self.manually_set_random and self.flip):
            if isinstance(clip[0], np.ndarray):
                if self.is_flow:
                    raise NotImplementedError("Inverting img in ndarray form is not implemented.")
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                # The Flow part referenced from
                # https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py
                return [
                    ImageOps.invert(img.transpose(PIL.Image.FLIP_LEFT_RIGHT))
                    if self.is_flow and i % 2 == 0 else
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    for i, img in enumerate(clip)
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image'
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class ResizeShorterSide(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (int): the vidoe will be resized sich taht the shorter side is this number.
    """

    def __init__(self, size=256, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        if im_w < im_h:
            scaling_factor = self.size / im_w
        else:
            scaling_factor = self.size / im_h

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest', multi_clips=False):
        self.size = size
        self.interpolation = interpolation
        self.multi_clips = multi_clips

    def __call__(self, clip):
        if self.multi_clips:
            return [self.work(c) for c in clip]
        else:
            return self.work(clip)

    def work(self, clip):
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image'
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds

    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.misc.imrotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image'
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image'
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomCenterCornerCrop(object):
    """Extract center crop or the corners for a list of images

    Args:
    size (sequence or int): Desired output size for th crop in format (h, w)
    size_options (list): The width or hight to crop will be chosen randomly
        from this options sequence if specified, and the 'size' argument will be ignored.
    """

    def __init__(self, size=0, size_options=[]):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size
        self.size_options = size_options

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        if len(self.size_options) == 0:
            h, w = self.size
        else:
            h = random.choice(self.size_options)
            w = random.choice(self.size_options)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image'
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        decision = random.randrange(0, 5)
        if decision == 0:
            # center crop
            x1 = int(round((im_w - w) / 2.))
            y1 = int(round((im_h - h) / 2.))
        elif decision == 1:
            x1, y1 = 0, 0  # left up corner
        elif decision == 2:
            x1, y1 = 0, im_h - h  # left down corner
        elif decision == 3:
            x1, y1 = im_w - w, 0  # right up corner
        elif decision == 4:
            x1, y1 = im_w - w, im_h - h  # right down corner
        else:
            raise ValueError('The random integer should be in the range [0, 4]')
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image

        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image'
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip
