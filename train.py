#!/usr/bin/env python

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

This code trains the NetVLAD neural network used to extract Patch-NetVLAD features.
'''


from __future__ import print_function

import argparse
import configparser
import os
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile, isdir
from os import makedirs
from datetime import datetime
import tempfile

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset

import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np

from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.tools.msls import ImagesFromList
from patchnetvlad.models.models_generic import get_backend, get_model, Flatten, L2Norm
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import trange, tqdm

from patchnetvlad.tools.msls import MSLS


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string"""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def train(epoch_num, config):
    train_dataset.new_epoch()

    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // int(config['train']['batchsize'])

    for subIter in trange(train_dataset.nCacheSubset, desc='Cache refresh'.rjust(15), position=1):
        pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])

        tqdm.write('====> Building Cache')
        train_dataset.update_subcache(model, pool_size)

        training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                          batch_size=int(config['train']['batchsize']), shuffle=True,
                                          collate_fn=MSLS.collate_fn, pin_memory=cuda)

        tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
        tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_cached()))

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in \
                enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            data_input = torch.cat([query, positives, negatives])

            data_input = data_input.to(device)
            image_encoding = model.encoder(data_input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])

            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del data_input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration,
                                                                       nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch_num - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch_num - 1) * nBatches) + iteration)
                tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
                tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_cached()))

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)


def test(eval_set, epoch_num=0, write_tboard=False, pbar_position=0):
    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform())
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform())
    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          num_workers=opt.threads, batch_size=int(config['train']['cachebatchsize']),
                                          shuffle=False, pin_memory=cuda)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                      num_workers=opt.threads, batch_size=int(config['train']['cachebatchsize']),
                                      shuffle=False, pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])
        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_encoding = model.encoder(input_data)

                vlad_encoding = model.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding

    del test_data_loader_queries, test_data_loader_dbs

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    # for each query get those within threshold distance
    gt = eval_set.all_pos_indices

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(eval_set.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)

    return all_recalls


def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    cluster_sampler = SubsetRandomSampler(np.random.choice(len(cluster_set.dbImages), nIm, replace=False))

    cluster_data_loader = DataLoader(dataset=ImagesFromList(cluster_set.dbImages, transform=input_transform()),
                                     num_workers=opt.threads, batch_size=int(config['train']['cachebatchsize']), shuffle=False,
                                     pin_memory=cuda,
                                     sampler=cluster_sampler)

    if not exists(join(opt.cache_path, 'centroids')):
        makedirs(join(opt.cache_path, 'centroids'))

    initcache_clusters = join(opt.cache_path, 'centroids',
                              'vgg16_' + opt.trainset + '_' + config['train']['numclusters'] + '_desc_cen.hdf5')
    with h5py.File(initcache_clusters, mode='w') as h5_file:
        with torch.no_grad():
            model.eval()
            tqdm.write('====> Extracting Descriptors')
            dbFeat = h5_file.create_dataset("descriptors", [nDescriptors, encoder_dim], dtype=np.float32)

            for iteration, (input_data, indices) in enumerate(tqdm(cluster_data_loader, desc='Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_descriptors = model.encoder(input_data).view(input_data.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration - 1) * int(config['train']['cachebatchsize']) * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix * nPerImage
                    dbFeat[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                del input_data, image_descriptors

        tqdm.write('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, int(config['train']['numclusters']), niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        tqdm.write('====> Storing centroids ' + str(kmeans.centroids.shape))
        h5_file.create_dataset('centroids', data=kmeans.centroids)
        tqdm.write('====> Done!')


def save_checkpoint(state, is_best_sofar, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.save_file_path, filename)
    torch.save(state, model_out_path)
    if is_best_sofar:
        shutil.copyfile(model_out_path, join(opt.save_file_path, 'model_best.pth.tar'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-train')

    parser.add_argument('--config_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--cache_path', type=str, default=tempfile.mkdtemp(),
                        help='Path to save cache, centroid data to.')
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save checkpoints to')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for resuming training.')
    parser.add_argument('--dataset_root_dir', type=str, default='/work/qvpr/data/raw/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--identifier', type=str, default='mapillary_nopanos',
                        help='Description of this model, e.g. mapillary_nopanos_vgg16_netvlad')

    parser.add_argument('--trainset', type=str, default='mapillary', help='Which training set to use', choices=['mapillary, pitts'])
    parser.add_argument('--includepanos', action='store_true', help='Train with panoramas included (only valid for mapillary)')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')


    opt = parser.parse_args()
    print(opt)

    # TODO: add some kind of check if a resumed ckpt has different parameters compared to the train.ini file
    # TODO: implement training on pitts

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    optimizer = None
    scheduler = None

    print('===> Building model')

    encoder_dim, encoder = get_backend()

    if opt.resume_path: # if already started training earlier and continuing
        if isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            model = get_model(encoder, encoder_dim, opt, config['global_params'], append_pca_layer=False)

            model.load_state_dict(checkpoint['state_dict'])
            opt.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(opt.resume_path, ))

            print('debugpoint')
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        resume_pca = False
        print('===> Finding cluster centroids')

        print('===> Loading dataset(s) for clustering')
        train_dataset = MSLS(opt.dataset_root_dir, mode='test', cities='train', transform=input_transform(),
                             bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                             margin=float(config['train']['margin']))

        print('===> Loading model for clustering')
        config['global_params']['num_clusters'] = config['train']['num_clusters']

        model = get_model(encoder, encoder_dim, opt, config['global_params'], append_pca_layer=False)

        model = model.to(device)

        print('===> Calculating descriptors and clusters')
        get_clusters(train_dataset)

        # a little hacky, but needed to easily run init_params
        model = model.to(device="cpu")

        initcache = join(opt.cache_path, 'centroids', 'vgg16_' + opt.trainset + '_' + config['train'][
                                      'numclusters'] + '_desc_cen.hdf5')
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pool.init_params(clsts, traindescs)
            del clsts, traindescs

        print('debugpoint')

    isParallel = False
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if config['train']['optim'] == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=float(config['train']['lr']))  # , betas=(0,0.9))
    elif config['train']['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']),
                              weight_decay=float(config['train']['weightDecay']))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['lrstep']),
                                              gamma=float(config['train']['lrgamma']))
    else:
        raise ValueError('Unknown optimizer: ' + config['train']['optim'])

    criterion = nn.TripletMarginLoss(margin=float(config['train']['margin']) ** 0.5, p=2, reduction='sum').to(device)

    model = model.to(device)

    if opt.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('===> Loading dataset(s)')
    exlude_panos_training = not opt.includepanos
    train_dataset = MSLS(opt.dataset_root_dir, mode='train', transform=input_transform(),
                         bs=int(config['train']['cachebatchsize']), threads=opt.threads, margin=float(config['train']['margin']),
                         exclude_panos=exlude_panos_training)

    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(),
                              bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                              margin=float(config['train']['margin']), posDistThr=25)

    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))
    print('===> Training model')
    writer = SummaryWriter(
        log_dir=join(opt.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.save_file_path = join(logdir, 'checkpoints')
    makedirs(opt.save_file_path)

    not_improved = 0
    best_score = 0
    if opt.resume_path:
        # noinspection PyUnboundLocalVariable
        if 'not_improved' in checkpoint:
            not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        train(epoch, config)
        if scheduler is not None:
            scheduler.step(epoch)
        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = test(validation_dataset, epoch, write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, is_best)

            if int(config['train']['patience']) > 0 and not_improved > (int(config['train']['patience']) / int(config['train']['evalevery'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')