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

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Initialises NetVLAD clusters before training.
'''


from math import ceil
from os.path import join, exists
from os import makedirs
import torch
import torch.nn.functional as F
import h5py
import faiss
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm
from patchnetvlad.training_tools.msls import ImagesFromList
from patchnetvlad.tools.datasets import input_transform


def get_clusters(cluster_set, model, encoder_dim, device, opt, config):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
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
                              'vgg16_' + 'mapillary_' + config['train']['num_clusters'] + '_desc_cen.hdf5')
    with h5py.File(initcache_clusters, mode='w') as h5_file:
        with torch.no_grad():
            model.eval()
            tqdm.write('====> Extracting Descriptors')
            dbFeat = h5_file.create_dataset("descriptors", [nDescriptors, encoder_dim], dtype=np.float32)

            for iteration, (input_data, indices) in enumerate(tqdm(cluster_data_loader, desc='Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_descriptors = model.encoder(input_data).view(input_data.size(0), encoder_dim, -1).permute(0, 2, 1)
                image_descriptors = F.normalize(image_descriptors, p=2, dim=2) # we L2-norm descriptors before vlad so
                # need to L2-norm here as well

                batchix = (iteration - 1) * int(config['train']['cachebatchsize']) * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix * nPerImage
                    dbFeat[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                del input_data, image_descriptors

        tqdm.write('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, int(config['train']['num_clusters']), niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        tqdm.write('====> Storing centroids ' + str(kmeans.centroids.shape))
        h5_file.create_dataset('centroids', data=kmeans.centroids)
        tqdm.write('====> Done!')