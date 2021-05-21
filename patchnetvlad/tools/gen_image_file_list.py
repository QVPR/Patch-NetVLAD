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
'''

from os.path import join
import glob
import argparse
from natsort import natsorted


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

parser = argparse.ArgumentParser(description='genImageList')
parser.add_argument('--dataset_root_dir', type=str,
                    help='Path to folder where the images are saved.')
parser.add_argument('--out_file', type=str,
                    help='File to which the image list should be saved to.')

if __name__ == "__main__":
    opt = parser.parse_args()
    image_list = []
    for fname in natsorted(glob.glob(opt.dataset_root_dir + '/*.*')):
        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
            path = join(opt.dataset_root_dir, fname)
            image_list.append(path)

    with open(opt.out_file, 'w') as out_file:
        for image_name in image_list:
            out_file.write(join(opt.dataset_root_dir, image_name) + '\n')

    print('Done')
