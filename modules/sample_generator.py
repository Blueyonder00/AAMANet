import numpy as np
from PIL import Image

from .utils import *


def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):
    """
    generator
    bbox: 首帧图像的bbox:[209. 155. 100.  63.]
    n: 以首帧图像的bbox选取的样本个数:1000
    overlap_range :[0.6, 1]
    scale_range: [1, 2]
    """
    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 128:
            samples_ = generator(bbox, remain * factor)  # samples_:(2000,4)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx, :]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor * 2

        return samples


class SampleGenerator:
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        """
        type:
        img_size:
        trans_f:0.1
        scale_f:1.2
        aspect_f:1.1
        """
        self.type = type
        self.img_size = np.array([img_size[1],img_size[0]]).reshape(1,-1)  # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid
        # self.std = std

    def __call__(self, bb, n):  # #16 call
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1)) # copy (center_x, center_y, w, h) in lengthways

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n, 1) * 2 - 1  # [0，1)x2-1 uniform distribution
            samples[:, 2:] *= self.aspect_f ** np.concatenate([ratio, -ratio], axis=1) # w**ratio, h**(-ratio)

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1) # vary center
            samples[:, 2:] *= self.scale_f ** np.clip(0.5 * np.random.randn(n, 1), -1, 1) # vary w,h



        # if self.type == 'specific_gaussian':
        #     samples[:, :1] += self.trans_f * np.mean(bb[2]) * 0.5 * np.random.normal(0,self.std[0],(n,2))
        #     samples[:, :2] += self.trans_f * np.mean(bb[3]) * 0.5 * np.random.normal(0, self.std[1], (n, 2))
        #     samples[:, 2:3] *= self.scale_f ** 0.5 * np.random.normal(0,self.std[2],(n,2))
        #     samples[:, 3:] *= self.scale_f ** 0.5*np.random.normal(0,self.std[3],(n,2))

        elif self.type == 'uniform':
            samples[:, :2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            # samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
            samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        # samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        samples[:, 2:] = np.clip(samples[:, 2:], 5, self.img_size - 5)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f
