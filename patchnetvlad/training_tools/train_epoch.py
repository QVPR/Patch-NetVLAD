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

Trains an epoch of NetVLAD, using the Mapillary Street-level Sequences Dataset.
'''


import torch
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from patchnetvlad.training_tools.tools import humanbytes
from patchnetvlad.training_tools.msls import MSLS


def train_epoch(train_dataset, model, optimizer, criterion, encoder_dim, device, epoch_num, opt, config, writer):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
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
