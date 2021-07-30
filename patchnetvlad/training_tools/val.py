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

Validation of NetVLAD, using the Mapillary Street-level Sequences Dataset.
'''


import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from patchnetvlad.training_tools.msls import ImagesFromList
from patchnetvlad.tools.datasets import input_transform


def val(eval_set, model, encoder_dim, device, opt, config, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
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