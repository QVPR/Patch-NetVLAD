#!/usr/bin/env python

'''
Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Extracts Patch-NetVLAD local and NetVLAD global features from a given directory of images.

Configuration settings are stored in configs folder, with compute heavy performance or light-weight alternatives
available.

Features are saved into a nominated output directory, with one file per image per patch size.

Code is dynamic and can be configured with essentially *any* number of patch sizes, by editing the config files.
'''


import argparse
import configparser
import os
from os.path import join, exists, isfile

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt


from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools.patch_matcher import PatchMatcher
from patchnetvlad.models.local_matcher import calc_keypoint_centers_from_patches as calc_keypoint_centers_from_patches
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR


def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def normalise_func(input_diff, num_patches, patch_weights):
    normed_diff = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        normed_diff = normed_diff + (patch_weights[i] * input_diff[i])
    return normed_diff


def plot_two(cv_im_one, cv_im_two, inlier_keypoints_one, inlier_keypoints_two, plot_save_path):
    # keypoint_colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    # from PIL import ImageColor
    # keypoint_colors = [ImageColor.getcolor(keypoint_color, "RGB") for keypoint_color in keypoint_colors]

    kp_all1 = []
    kp_all2 = []
    matches_all = []
    for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(inlier_keypoints_one, inlier_keypoints_two):
        for i in range(this_inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(this_inlier_keypoints_one[i, 0].astype(float), this_inlier_keypoints_one[i, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(this_inlier_keypoints_two[i, 0].astype(float), this_inlier_keypoints_two[i, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(i, i, 0))

    im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2,
                                          matches_all, None, matchColor=(0, 255, 0), flags=2)

    im_allpatch_matches = cv2.cvtColor(im_allpatch_matches, cv2.COLOR_BGR2RGB)

    plt.imshow(im_allpatch_matches)
    # plt.show()
    plt.axis('off')
    filename = join(plot_save_path, 'patchMatchings.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def match_two(model, device, opt, config):

    pool_size = int(config['global_params']['num_pcs'])

    model.eval()

    im_one = Image.open(opt.first_im_path)
    im_two = Image.open(opt.second_im_path)

    it = input_transform((int(config['feature_extract']['imageresizeH']), int(config['feature_extract']['imageresizeW'])))

    im_one = it(im_one).unsqueeze(0)
    im_two = it(im_two).unsqueeze(0)

    input_data = torch.cat((im_one.to(device), im_two.to(device)), 0)

    tqdm.write('====> Extracting Features')
    with torch.no_grad():
        image_encoding = model.encoder(input_data)

        vlad_local, _ = model.pool(image_encoding)
        # global_feats = get_pca_encoding(model, vlad_global).cpu().numpy()

        local_feats_one = []
        local_feats_two = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
            local_feats_two.append(this_local_feats[1, :, :])

    tqdm.write('====> Calculating Keypoint Positions')
    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    tqdm.write('====> Matching Local Features')
    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)

    scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
    score = -normalise_func(scores, len(patch_sizes), patch_weights)

    print(f"Similarity score between the two images is: {score:.5f}. Larger scores indicate better matches.")

    if config['feature_match']['matcher'] == 'RANSAC':
        tqdm.write('====> Plotting Local Features and save them to ' + str(join(opt.plot_save_path, 'patchMatchings.png')))

        # using cv2 for their in-built keypoint correspondence plotting tools
        cv_im_one = cv2.imread(opt.first_im_path, -1)
        cv_im_two = cv2.imread(opt.second_im_path, -1)
        cv_im_one = cv2.resize(cv_im_one, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
        cv_im_two = cv2.resize(cv_im_two, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
        # cv2 resize slightly different from torch, but for visualisation only not a big problem

        plot_two(cv_im_one, cv_im_two, inlier_keypoints_one, inlier_keypoints_two, opt.plot_save_path)


def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Match-Two')
    parser.add_argument('--config_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'configs/performance.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--first_im_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'example_images/tokyo_db.png'),
                        help='Full path (with extension) to an image file')
    parser.add_argument('--second_im_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'example_images/tokyo_query.jpg'),
                        help='Full path (with extension) to another image file')
    parser.add_argument('--plot_save_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'results'),
                        help='Path plus optional prefix pointing to a location to save the output matching plot')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    # backup: try whether resume_ckpt is relative to script path
    if not isfile(resume_ckpt):
        resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
        if not isfile(resume_ckpt):
            from download_models import download_all_models
            download_all_models(ask_for_permission=True)

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, opt, config['global_params'], append_pca_layer=True)

        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    match_two(model, device, opt, config)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('Done')

if __name__ == "__main__":
    main()
