#!/usr/bin/env python

import os
import urllib.request
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR


def ask_yesno(question):
    """
    Helper to get yes / no answer from user.
    """
    yes = {'yes', 'y'}
    no = {'no', 'n', 'q', 'quit'}  # pylint: disable=invalid-name

    done = False
    print(question)
    while not done:
        choice = input().lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond \'yes\' or \'no\'.")


def download_all_models(ask_for_permission=False):
    dest_dir = os.path.join(PATCHNETVLAD_ROOT_DIR, 'pretrained_models')
    if not ask_for_permission or ask_yesno("Auto-download pretrained models into " + dest_dir + " (takes around 2GB of space)? Yes/no."):
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar")):
            print('Downloading mapillary_WPCA128.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/vvr0jizjti0z2LR/download", os.path.join(dest_dir, "mapillary_WPCA128.pth.tar"))
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar")):
            print('Downloading mapillary_WPCA512.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/DFxbGgFwh1y1wAz/download", os.path.join(dest_dir, "mapillary_WPCA512.pth.tar"))
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar")):
            print('Downloading mapillary_WPCA4096.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/ZgW7DMEpeS47ELI/download", os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar"))
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar")):
            print('Downloading pittsburgh_WPCA128.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/2ORvaCckitjz4Sd/download", os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar"))
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar")):
            print('Downloading pittsburgh_WPCA512.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/WKl45MoboSyB4SH/download", os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar"))
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar")):
            print('Downloading pittsburgh_WPCA4096.pth.tar')
            urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/1aoTGbFjsekeKlB/download", os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar"))
        print('Downloaded all pretrained models.')

if __name__ == "__main__":
    download_all_models()
