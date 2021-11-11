'''
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

Copyright (c) 2020 mapillary

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

Modified by Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

'''


import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm


default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [Image.open(im) for im in self.images[idx].split(",")]
        except:
            img = [Image.open(self.images[0])]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx


class MSLS(Dataset):
    def __init__(self, root_dir, cities='', nNeg=5, transform=None, mode='train', task='im2im', subtask='all',
                 seq_length=1, posDistThr=10, negDistThr=25, cached_queries=1000, cached_negatives=1000,
                 positive_sampling=True, bs=24, threads=8, margin=0.1, exclude_panos=True):

        # initializing
        assert mode in ('train', 'val', 'test')
        assert task in ('im2im', 'im2seq', 'seq2im', 'seq2seq')
        assert subtask in ('all', 's2w', 'w2s', 'o2n', 'n2o', 'd2n', 'n2d')
        assert seq_length % 2 == 1
        assert (task == 'im2im' and seq_length == 1) or (task != 'im2im' and seq_length > 1)

        if cities in default_cities:
            self.cities = default_cities[cities]
        elif cities == '':
            self.cities = default_cities[mode]
        else:
            self.cities = cities.split(',')

        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        self.sideways = []
        self.night = []
        self.qEndPosList = []
        self.dbEndPosList = []

        self.all_pos_indices = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.exclude_panos = exclude_panos
        self.mode = mode
        self.subtask = subtask
        print('Exclude panoramas:', self.exclude_panos)

        # other
        self.transform = transform

        # define sequence length based on task
        if task == 'im2im':
            seq_length_q, seq_length_db = 1, 1
        elif task == 'seq2seq':
            seq_length_q, seq_length_db = seq_length, seq_length
        elif task == 'seq2im':
            seq_length_q, seq_length_db = seq_length, 1
        else:  # im2seq
            seq_length_q, seq_length_db = 1, seq_length

        # load data
        for city in self.cities:
            print("=====> {}".format(city))

            subdir = 'test' if city in default_cities['test'] else 'train_val'

            # get len of images from cities so far for indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)

            # when GPS / UTM is available
            if self.mode in ['train', 'val']:
                # load query data
                qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col=0)
                qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                # load database data
                dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col=0)
                dbDataRaw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)

                # arange based on task
                qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir, city, 'query'), seq_length_q)
                dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'),
                                                          seq_length_db)

                # filter based on subtasks
                if self.mode in ['val']:
                    qIdx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                    dbIdx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                    # find all the sequence where the center frame belongs to a subtask
                    val_frames = np.where(qIdx[self.subtask])[0]
                    qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, val_frames)

                    val_frames = np.where(dbIdx[self.subtask])[0]
                    dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, val_frames)

                # filter based on panorama data
                if self.exclude_panos:
                    panos_frames = np.where((qDataRaw['pano'] == False).values)[0]
                    qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, panos_frames)

                    panos_frames = np.where((dbDataRaw['pano'] == False).values)[0]
                    dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, panos_frames)

                unique_qSeqIdx = np.unique(qSeqIdxs)
                unique_dbSeqIdx = np.unique(dbSeqIdxs)

                # if a combination of city, task and subtask is chosen, where there are no query/dabase images,
                # then continue to next city
                if len(unique_qSeqIdx) == 0 or len(unique_dbSeqIdx) == 0:
                    continue

                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)

                self.qEndPosList.append(len(qSeqKeys))
                self.dbEndPosList.append(len(dbSeqKeys))

                qData = qData.loc[unique_qSeqIdx]
                dbData = dbData.loc[unique_dbSeqIdx]

                # useful indexing functions
                seqIdx2frameIdx = lambda seqIdx, seqIdxs: seqIdxs[seqIdx]
                # frameIdx2seqIdx = lambda frameIdx, seqIdxs: np.where(seqIdxs == frameIdx)[0][1]
                frameIdx2uniqFrameIdx = lambda frameIdx, uniqFrameIdx: np.where(np.in1d(uniqFrameIdx, frameIdx))[0]
                uniqFrameIdx2seqIdx = lambda frameIdxs, seqIdxs: \
                    np.where(np.in1d(seqIdxs, frameIdxs).reshape(seqIdxs.shape))[0]

                # utm coordinates
                utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
                utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)

                night, sideways, index = qData['night'].values, (
                            qData['view_direction'] == 'Sideways').values, qData.index

                # find positive images for training
                neigh = NearestNeighbors(algorithm='brute')
                neigh.fit(utmDb)
                pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
                self.all_pos_indices.extend(pos_indices)

                if self.mode == 'train':
                    nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                for q_seq_idx in range(len(qSeqKeys)):

                    q_frame_idxs = seqIdx2frameIdx(q_seq_idx, qSeqIdxs)
                    q_uniq_frame_idx = frameIdx2uniqFrameIdx(q_frame_idxs, unique_qSeqIdx)

                    p_uniq_frame_idxs = np.unique([p for pos in pos_indices[q_uniq_frame_idx] for p in pos])

                    # the query image has at least one positive
                    if len(p_uniq_frame_idxs) > 0:
                        p_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[p_uniq_frame_idxs], dbSeqIdxs))

                        self.pIdx.append(p_seq_idx + _lenDb)
                        self.qIdx.append(q_seq_idx + _lenQ)

                        # in training we have two thresholds, one for finding positives and one for finding images
                        # that we are certain are negatives.
                        if self.mode == 'train':

                            n_uniq_frame_idxs = np.unique([n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg])
                            n_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[n_uniq_frame_idxs], dbSeqIdxs))

                            self.nonNegIdx.append(n_seq_idx + _lenDb)

                            # gather meta which is useful for positive sampling
                            if sum(night[np.in1d(index, q_frame_idxs)]) > 0:
                                self.night.append(len(self.qIdx) - 1)
                            if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0:
                                self.sideways.append(len(self.qIdx) - 1)

            # when GPS / UTM / pano info is not available
            elif self.mode in ['test']:

                # load images for subtask
                qIdx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                dbIdx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                # arange in sequences
                qSeqKeys, qSeqIdxs = self.arange_as_seq(qIdx, join(root_dir, subdir, city, 'query'), seq_length_q)
                dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbIdx, join(root_dir, subdir, city, 'database'),
                                                          seq_length_db)

                # filter query based on subtask
                val_frames = np.where(qIdx[self.subtask])[0]
                qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, val_frames)

                # filter database based on subtask
                val_frames = np.where(dbIdx[self.subtask])[0]
                dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, val_frames)

                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)

                # add query index
                self.qIdx.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))

                # if a combination of cities, task and subtask is chosen, where there are no query/database images,
                # then exit
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print(
                "A combination of cities, task and subtask have been chosen, where there are no query/database images.")
            print("Try choosing a different subtask or more cities")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx)
        self.dbImages = np.asarray(self.dbImages)
        self.sideways = np.asarray(self.sideways)
        self.night = np.asarray(self.night)

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threads = threads
        self.bs = bs

        if mode == 'train':

            # for now always 1-1 lookup.
            self.negCache = np.asarray([np.empty((0,), dtype=int)] * len(self.qIdx))

            # calculate weights for positive sampling
            if positive_sampling:
                self.__calcSamplingWeights__()
            else:
                self.weights = np.ones(len(self.qIdx)) / float(len(self.qIdx))

    def __calcSamplingWeights__(self):

        # length of query
        N = len(self.qIdx)

        # initialize weights
        self.weights = np.ones(N)

        # weight higher if from night or sideways facing
        if len(self.night) != 0:
            self.weights[self.night] += N / len(self.night)
        if len(self.sideways) != 0:
            self.weights[self.sideways] += N / len(self.sideways)

        # print weight information
        print("#Sideways [{}/{}]; #Night; [{}/{}]".format(len(self.sideways), N, len(self.night), N))
        print("Forward and Day weighted with {:.4f}".format(1))
        if len(self.night) != 0:
            print("Forward and Night weighted with {:.4f}".format(1 + N / len(self.night)))
        if len(self.sideways) != 0:
            print("Sideways and Day weighted with {:.4f}".format(1 + N / len(self.sideways)))
        if len(self.sideways) != 0 and len(self.night) != 0:
            print("Sideways and Night weighted with {:.4f}".format(1 + N / len(self.night) + N / len(self.sideways)))

    @staticmethod
    def arange_as_seq(data, path, seq_length):

        seqInfo = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        seq_keys, seq_idxs = [], []
        for idx in data.index:

            # edge cases.
            if idx < (seq_length // 2) or idx >= (len(seqInfo) - seq_length // 2):
                continue

            # find surrounding frames in sequence
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            seq = seqInfo.iloc[seq_idx]

            # the sequence must have the same sequence key and must have consecutive frames
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                seq_keys.append(seq_key)
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    @staticmethod
    def filter(seqKeys, seqIdxs, center_frame_condition):
        keys, idxs = [], []
        for key, idx in zip(seqKeys, seqIdxs):
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, negCounts, indices

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, outputdim=None):

        # reset triplets
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        if net is None:
            qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=False)

            for q in qidxs:

                # get query idx
                qidx = self.qIdx[q]

                # get positives
                pidxs = self.pIdx[q]

                # choose a random positive (within positive range (default 10 m))
                pidx = np.random.choice(pidxs, size=1)[0]

                # get negatives
                while True:
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)

                    # ensure that non of the choice negative images are within the negative range (default 25 m)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break

                # package the triplet and target
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)

                self.triplets.append((triplet, target))

            # increment subset counter
            self.current_subset += 1

            return

        # take n query images
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.transform), **opt)
        nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.transform), **opt)

        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device)
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device)
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                qvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(ploader), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                pvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(nloader), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                nvecs[i * bs:(i + 1) * bs, :] = vlad_encoding

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

    def __getitem__(self, idx):
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        query = self.transform(Image.open(self.qImages[qidx]))
        positive = self.transform(Image.open(self.dbImages[pidx]))
        negatives = [self.transform(Image.open(self.dbImages[idx])) for idx in nidx]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qidx, pidx] + nidx