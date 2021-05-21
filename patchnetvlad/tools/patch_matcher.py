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

This file is designed to house a variety of different types of matchers, all
easily callable by a new matcher class. It is easy to add your own new matcher,
just add a new method and add an additional elif to def match(...).
'''

import numpy as np
import torch
import cv2


def torch_nn(x, y):
    mul = torch.matmul(x, y)

    dist = 2 - 2 * mul + 1e-9
    dist = torch.sqrt(dist)

    _, fw_inds = torch.min(dist, 0)
    bw_inds = torch.argmin(dist, 1)

    return fw_inds, bw_inds


class PatchMatcher(object):
    def __init__(self, matcher, patch_sizes, strides, all_keypoints, all_indices):
        self.matcher = matcher
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.all_keypoints = all_keypoints
        self.all_indices = all_indices

    def match(self, qfeats, dbfeats):
        if self.matcher == 'RANSAC':
            return self.compare_two_ransac(qfeats, dbfeats)
        elif self.matcher == 'spatialApproximator':
            return self.compare_two_spatial(qfeats, dbfeats)
        else:
            raise ValueError('unknown matcher descriptor')

    def compare_two_ransac(self, qfeats, dbfeats):
        scores = []
        all_inlier_index_keypoints = []
        all_inlier_query_keypoints = []
        for qfeat, dbfeat, keypoints, stride in zip(qfeats, dbfeats, self.all_keypoints, self.strides):
            fw_inds, bw_inds = torch_nn(qfeat, dbfeat)

            fw_inds = fw_inds.cpu().numpy()
            bw_inds = bw_inds.cpu().numpy()

            mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

            if len(mutuals) > 3: # need at least four points to estimate a Homography
                index_keypoints = keypoints[:, mutuals]
                query_keypoints = keypoints[:, fw_inds[mutuals]]

                index_keypoints = np.transpose(index_keypoints)
                query_keypoints = np.transpose(query_keypoints)

                _, mask = cv2.findHomography(index_keypoints, query_keypoints, cv2.FM_RANSAC,
                                             ransacReprojThreshold=16*stride*1.5)
                # RANSAC reproj threshold is set to the (stride*1.5) in image space for vgg-16, given a particular patch stride
                # in this work, we ignore the H matrix output - but users of this code are welcome to utilise this for
                # pose estimation (something we may also investigate in future work)

                inlier_index_keypoints = index_keypoints[mask.ravel() == 1]
                all_inlier_query_keypoints.append(query_keypoints[mask.ravel() == 1])
                inlier_count = inlier_index_keypoints.shape[0]
                scores.append(-inlier_count / qfeat.shape[0])
                all_inlier_index_keypoints.append(inlier_index_keypoints)
                # we flip to negative such that best match is the smallest number, to be consistent with vanilla NetVlad
                # we normalise by patch count to remove biases in the scoring between different patch sizes (so that all
                # patch sizes are weighted equally and that the only weighting is from the user-defined patch weights)
            else:
                scores.append(0.)

        return scores, all_inlier_query_keypoints, all_inlier_index_keypoints

    def compare_two_spatial(self, qfeats, dbfeats):
        scores = []
        for qfeat, dbfeat, indices in zip(qfeats, dbfeats, self.all_indices):
            fw_inds, bw_inds = torch_nn(qfeat, dbfeat)

            fw_inds = fw_inds.cpu().numpy()
            bw_inds = bw_inds.cpu().numpy()

            mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

            if len(mutuals) > 0:
                index_keypoints = indices[:, mutuals]
                query_keypoints = indices[:, fw_inds[mutuals]]

                spatial_dist = index_keypoints - query_keypoints # manhattan distance works reasonably well and is fast
                mean_spatial_dist = np.mean(spatial_dist, axis=1)

                # residual between a spatial distance and the mean spatial distance. Smaller is better
                s_dists_x = spatial_dist[0, :] - mean_spatial_dist[0]
                s_dists_y = spatial_dist[1, :] - mean_spatial_dist[1]
                s_dists_x = np.absolute(s_dists_x)
                s_dists_y = np.absolute(s_dists_y)

                # anchor to the maximum x and y axis index for the patch "feature space"
                xmax = np.max(indices[0, :])
                ymax = np.max(indices[1, :])

                # find second-order residual, by comparing the first residual to the respective anchors
                # after this step, larger is now better
                # add non-linearity to the system to excessively penalise deviations from the mean
                s_score = (xmax - s_dists_x)**2 + (ymax - s_dists_y)**2

                scores.append(-s_score.sum() / qfeat.shape[0])
                # we flip to negative such that best match is the smallest number, to be consistent with vanilla NetVlad
                # we normalise by patch count to remove biases in the scoring between different patch sizes (so that all
                # patch sizes are weighted equally and that the only weighting is from the user-defined patch weights)
            else:
                scores.append(0.)

        return scores, None, None
