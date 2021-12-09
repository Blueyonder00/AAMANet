import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0, '../modules')
from modules.sample_generator import *
from modules.utils import *


class RegionDataset(data.Dataset):
    def __init__(self, rgb_img_dir, rgb_img_list, t_img_dir, t_img_list, gt, opts):
        self.rgb_img_list=[]
        self.t_img_list=[]
        for i in range(3):
            self.rgb_img_list.append(np.array([os.path.join(rgb_img_dir[i], img) for img in rgb_img_list]))
            self.t_img_list.append(np.array([os.path.join(t_img_dir[i], img) for img in t_img_list]))

        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.rgb_img_list[0]))
        self.pointer = 0
        image = np.empty([217,409,0])
        for j in range(3):
            single_Image = Image.open(self.rgb_img_list[j][0])
            single_Image = np.asarray(single_Image)
            image = np.dstack((image,single_Image))

        self.pos_generator = SampleGenerator('gaussian', image.shape[:2], 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.shape[:2], 1, 1.2, 1.1, True)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.rgb_img_list[0]))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.rgb_img_list[0]))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions1 = np.empty((0, 3, self.crop_size, self.crop_size))
        neg_regions1 = np.empty((0, 3, self.crop_size, self.crop_size))
        pos_regions2 = np.empty((0, 3, self.crop_size, self.crop_size))
        neg_regions2 = np.empty((0, 3, self.crop_size, self.crop_size))

        for i, bbox in enumerate(self.gt[idx]):
            image_rgb = np.empty([217,409,0])
            image_t = np.empty([217,409,0])
            for a in range(3):
                rgb_img_path = self.rgb_img_list[a][idx[i]]
                image_rgb_single = Image.open(rgb_img_path)
                image_rgb_single = np.asarray(image_rgb_single)
                image_rgb = np.dstack((image_rgb, image_rgb_single))
            for b in range(3):
                t_img_single = self.t_img_list[b][idx[i]]
                image_t_single = Image.open(t_img_single)
                image_t_single = np.asarray(image_t_single)
                image_t = np.dstack((image_t, image_t_single))

            n_pos = (self.batch_pos - len(pos_regions1)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions1)) // (self.batch_frames - i)
            pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(self.neg_generator, bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions1 = np.concatenate((pos_regions1, self.extract_regions(image_rgb, pos_examples)), axis=0)
            neg_regions1 = np.concatenate((neg_regions1, self.extract_regions(image_rgb, neg_examples)), axis=0)
            pos_regions2 = np.concatenate((pos_regions2, self.extract_regions(image_t, pos_examples)), axis=0)
            neg_regions2 = np.concatenate((neg_regions2, self.extract_regions(image_t, neg_examples)), axis=0)
        idx1 = idx
        idx2 = idx
        pos_regions1 = torch.from_numpy(pos_regions1).float()
        neg_regions1 = torch.from_numpy(neg_regions1).float()
        pos_regions2 = torch.from_numpy(pos_regions2).float()
        neg_regions2 = torch.from_numpy(neg_regions2).float()
        return pos_regions1, neg_regions1, idx1, pos_regions2, neg_regions2, idx2

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose((0, 3, 1, 2))
        regions = regions.astype('float32') - 128.
        return regions


##########################################################


# class RegionDataset1(data.Dataset):
#     def __init__(self, img_dir, img_list, gt, pos_regions, neg_regions, pos_examples, neg_examples, idx, opts):
#
#         self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
#         self.gt = gt
#
#         self.batch_frames = opts['batch_frames']
#         self.batch_pos = opts['batch_pos']
#         self.batch_neg = opts['batch_neg']
#
#         self.overlap_pos = opts['overlap_pos']
#         self.overlap_neg = opts['overlap_neg']
#
#         self.crop_size = opts['img_size']
#         self.padding = opts['padding']
#         self.idex = idx
#         self.pos_regions = pos_regions
#         self.neg_regions = neg_regions
#
#         image = Image.open(self.img_list[0]).convert('RGB')
#         self.pos_examples = pos_examples
#         self.neg_examples = neg_examples
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):

#         idx1 = idx
#         idx2 = idx1
#
#         pos_regions2 = np.empty((0, 3, self.crop_size, self.crop_size))
#         neg_regions2 = np.empty((0, 3, self.crop_size, self.crop_size))
#         for i, (img_path, bbox) in enumerate(zip(self.img_list[idx1], self.gt[idx1])):
#             image = Image.open(img_path).convert('RGB')
#             image = np.asarray(image)
#
#             pos_regions1 = self.pos_regions
#             neg_regions1 = self.neg_regions
#
#             pos_examples1 = self.pos_examples
#             neg_examples1 = self.neg_examples
#
#             pos_regions2 = np.concatenate((pos_regions2, self.extract_regions(image, pos_examples1)), axis=0)
#             neg_regions2 = np.concatenate((neg_regions2, self.extract_regions(image, neg_examples1)), axis=0)
#
#         pos_examples1 = torch.from_numpy(pos_examples1).float()
#         neg_examples1 = torch.from_numpy(neg_examples1).float()
#         pos_examples2 = pos_examples1
#         neg_examples2 = neg_examples1
#
#         pos_regions2 = torch.from_numpy(pos_regions2).float()
#         neg_regions2 = torch.from_numpy(neg_regions2).float()
#         return pos_regions1, neg_regions1, pos_examples1, neg_examples1, idx1, pos_regions2, neg_regions2, pos_examples2, neg_examples2, idx2
#
#     next1 = __next__
#
#     def extract_regions(self, image, samples):
#         regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
#         for i, sample in enumerate(samples):
#             regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
#
#         regions = regions.transpose((0, 3, 1, 2))
#         regions = regions.astype('float32') - 128.
#         return regions
