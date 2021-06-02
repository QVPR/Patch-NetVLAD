'''
MIT License

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

Receptive box calculations modified from tensorflow models under Apache License:

Copyright 2017 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import numpy as np
import torch
from tqdm.auto import tqdm
from patchnetvlad.tools.patch_matcher import PatchMatcher


def calc_receptive_boxes(height, width):
    """Calculate receptive boxes for each feature point.
    Modified from
    https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/delf/delf/python/feature_extractor.py

    Args:
      height: The height of feature map.
      width: The width of feature map.
      rf: The receptive field size.
      stride: The effective stride between two adjacent feature points.
      padding: The effective padding size.

    Returns:
      rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
      Each box is represented by [ymin, xmin, ymax, xmax].
    """

    rf, stride, padding = [196.0, 16.0, 90.0]  # hardcoded for vgg-16 conv5_3

    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    coordinates = torch.reshape(torch.stack([y, x], dim=2), [-1, 2])
    # [y,x,y,x]
    point_boxes = torch.cat([coordinates, coordinates], 1)
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    rf_boxes = stride * point_boxes + torch.FloatTensor(bias)
    return rf_boxes


def calc_keypoint_centers_from_patches(config, patch_size_h, patch_size_w, stride_h, stride_w):
    '''Calculate patch positions in image space

    Args:
        config: feature_match config data
        patch_size_h: height of patches
        patch_size_w: width of patches
        stride_h: stride in vertical direction between patches
        stride_w: stride in horizontal direction between patches

    :returns
        keypoints: patch positions back in image space for RANSAC
        indices: 2-D patch positions for rapid spatial scoring
    '''

    H = int(int(config['imageresizeH']) / 16)  # 16 is the vgg scaling from image space to feature space (conv5)
    W = int(int(config['imageresizeW']) / 16)
    padding_size = [0, 0]
    patch_size = (int(patch_size_h), int(patch_size_w))
    stride = (int(stride_h), int(stride_w))

    Hout = int((H + (2 * padding_size[0]) - patch_size[0]) / stride[0] + 1)
    Wout = int((W + (2 * padding_size[1]) - patch_size[1]) / stride[1] + 1)

    boxes = calc_receptive_boxes(H, W)

    num_regions = Hout * Wout

    k = 0
    indices = np.zeros((2, num_regions), dtype=int)
    keypoints = np.zeros((2, num_regions), dtype=int)
    # Assuming sensible values for stride here, may get errors with large stride values
    for i in range(0, Hout, stride_h):
        for j in range(0, Wout, stride_w):
            keypoints[0, k] = ((boxes[j + (i * W), 0] + boxes[(j + (patch_size[1] - 1)) + (i * W), 2]) / 2)
            keypoints[1, k] = ((boxes[j + ((i + 1) * W), 1] + boxes[j + ((i + (patch_size[0] - 1)) * W), 3]) / 2)
            indices[0, k] = j
            indices[1, k] = i
            k += 1

    return keypoints, indices


def normalise_func(input_diff, num_patches, patch_weights):
    normed_diff = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        normed_diff = normed_diff + patch_weights[i] * (
            (input_diff[:, i] - np.mean(input_diff[:, i])) / np.std(input_diff[:, i]))
    return normed_diff


def local_matcher(predictions, eval_set, input_query_local_features_prefix,
                  input_index_local_features_prefix, config, device):

    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    reordered_preds = []

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)

    for q_idx, pred in enumerate(tqdm(predictions, leave=False, desc='Patch compare pred')):
        diffs = np.zeros((predictions.shape[1], len(patch_sizes)))
        image_name_query = os.path.splitext(os.path.basename(eval_set.images[eval_set.numDb + q_idx]))[0]
        qfeat = []
        for patch_size in patch_sizes:
            qfilename = input_query_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_query + '.npy'
            qfeat.append(torch.transpose(torch.tensor(np.load(qfilename), device=device), 0, 1))
            # we pre-transpose here to save compute speed
        for k, candidate in enumerate(pred):
            image_name_index = os.path.splitext(os.path.basename(eval_set.images[candidate]))[0]
            dbfeat = []
            for patch_size in patch_sizes:
                dbfilename = input_index_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_index + '.npy'
                dbfeat.append(torch.tensor(np.load(dbfilename), device=device))

            diffs[k, :], _, _ = matcher.match(qfeat, dbfeat)

        diffs = normalise_func(diffs, len(patch_sizes), patch_weights)
        cand_sorted = np.argsort(diffs)
        reordered_preds.append(pred[cand_sorted])

    return reordered_preds
