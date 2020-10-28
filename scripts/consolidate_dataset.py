#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import math
import numpy as np
import argparse

from posthoc_learn.config import posthoc_config as config
from posthoc_learn.conban_dataset import ConBanDataset

def main(name, visual_model, haptic_model):
    dataset = ConBanDataset(name, visual_model, haptic_model, False)
    print()
    
    print("Exporting npz...")
    dataset.export()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gpu_id', default=config.gpu_id,
                    help="target gpu index to run this model")
    ap.add_argument('-n', '--name', default=config.data_name,
                    type=str, help="name of dataset to convert")
    ap.add_argument('-V', '--visual', default=config.visual_model,
                    type=str, help="Visual (SPANet) checkpoint")
    ap.add_argument('-H', '--haptic', default=config.haptic_model,
                    type=str, help="Haptic (HapticNet) checkpoint")
    args = ap.parse_args()

    if args.gpu_id == '-1':
        config.use_cuda = False
    else:
        config.use_cuda = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    main(args.name, args.visual, args.haptic)