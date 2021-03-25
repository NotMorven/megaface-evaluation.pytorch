#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# @Author   : Chengzhi.jcz
# @Email    : morven126@163.com
# @Site     : https://github.com/NotMorven
# @FileName : gen_megaface.py
# @Project  : megaface-evaluation.pytorch
# @Time     : 2021/1/10 下午9:20
# @Software : PyCharm
-----------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
from PIL import Image
from tqdm import tqdm

# import mxnet as mx
# from mxnet import ndarray as nd
import torch
import torch.nn.functional as F
from torchvision import transforms
# import model here
from models.sphere20a import sphere20a


data_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # faceNormToTensor(mean=128, std=128)  # custom pre-processing layer, but it's unnecessary
    transforms.Resize(size=(112, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


"""
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')  # delete A-channel
"""
# adapted from pytorch official
def read_img(image_path, mode='pil'):
    if mode == 'pil':  # pil format is adopted by pytorch official
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # delete A-channel
    # TODO: Not Implemented
    else:
        raise NotImplementedError("cv2 formatted data for pytorch is not Implemented in func(get_feature) yet")
        img = cv2.imread(image_path)
        return img


def get_feature(imgs, nets, device):
    count = len(imgs)
    # image_shape = [int(x) for x in args.image_size.split(',')]
    # data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[0], imgs[0].shape[1]))
    # data = torch.zeros((count * 2, 3, imgs[0].size[1], imgs[0].size[0]))  # imgs is list of PIL data
    data = torch.zeros((count * 2, 3, 112, 96))  # imgs is list of PIL data
    for idx, img in enumerate(imgs):
        # pil formatted data for pytorch
        for flipid in [0, 0]:
            if flipid == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_tensor = data_transforms(img)  # return tensor in shape(c, h. w)
            # rs_tensor = transforms.functional.resize(img=img_tensor, size=(112, 96))
            # rs_tensor = torch.nn.functional.interpolate()
            data[count * flipid + idx] = img_tensor  # tensor data in shape(n, c, h, w)

    data = data.to(device)

        # # this block is for cv2 format and mxnet
        # img = img[:, :, ::-1]  # to rgb
        # img = np.transpose(img, (2, 0, 1))
        # for flipid in [0, 1]:
        #     _img = np.copy(img)
        #     if flipid == 1:
        #         _img = _img[:, :, ::-1]
        #     _img = nd.array(_img)
        #     data[count * flipid + idx] = _img

    F = []
    for net in nets:
        # db = mx.io.DataBatch(data=(data,))
        # net.model.forward(db, is_train=False)
        # x = net.model.get_outputs()[0].asnumpy()

        with torch.no_grad():
            x = net(data).detach().cpu().numpy()
        embedding = x[0:count, :] + x[count:, :]
        embedding = sklearn.preprocessing.normalize(embedding)
        # print('emb', embedding.shape)
        F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = sklearn.preprocessing.normalize(F)
    # print('F', F.shape)
    return F


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, nets, device):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, nets, device)
    # print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)


def main(args):
    print(args)

    # device initial
    gpuid = args.gpu
    # ctx = mx.gpu(gpuid)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda", gpuid)

    # model init
    nets = []
    image_shape = [int(x) for x in args.image_size.split(',')]
    for model in args.model.split('|'):  # args.model is the model path

        net = sphere20a().to(device)
        net_state = torch.load(model, map_location=device)
        net.load_state_dict(net_state['model'], strict=False)
        net.eval()
        nets.append(net)

        # vec = model.split(',')
        # assert len(vec) > 1
        # prefix = vec[0]
        # epoch = int(vec[1])
        # print('loading', prefix, epoch)
        # net = edict()
        # net.ctx = ctx
        # net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
        # all_layers = net.sym.get_internals()
        # net.sym = all_layers['fc1_output']
        # net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
        # net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
        # net.model.set_params(net.arg_params, net.aux_params)
        # nets.append(net)

    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_out = os.path.join(args.output, 'megaface')

    i = 0
    succ = 0
    buffer = []
    pbar = tqdm(open(args.facescrub_lst, 'r'))
    for line in pbar:  ##############
        if i % 1000 == 0:
            # print("writing fs", i, succ)
            pbar.set_description("writing fs, i:{}, succ:{}".format(i, succ))           ###################
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]
        out_dir = os.path.join(facescrub_out, a)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(args.facescrub_root, image_path)
        img = read_img(image_path)  # in cv2 format
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets, device)  # extract feature
            buffer = []
        succ += 1

    if len(buffer) > 0:
        get_and_write(buffer, nets, device)
        buffer = []
    print('fs stat', i, succ)

    i = 0
    succ = 0
    buffer = []
    pbar_mega = tqdm(open(args.megaface_lst, 'r'))
    for line in pbar_mega:
        if i % 1000 == 0:
            # print("writing mf", i, succ)
            pbar.set_description("writing fs, i:{}, succ:{}".format(i, succ))  ###################
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # continue
        # print(landmark)
        image_path = os.path.join(args.megaface_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets, device)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, nets, device)
        buffer = []
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=128)
    parser.add_argument('--image_size', type=str, help='', default='3,112,96')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--algo', type=str, help='', default='LMC')
    parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
    parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
    parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
    parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
    parser.add_argument('--output', type=str, help='', default='./feature_out')
    parser.add_argument('--model', type=str, help='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

