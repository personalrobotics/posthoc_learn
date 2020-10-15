from __future__ import division
from __future__ import print_function

import sys
import os
from pathlib import Path
import yaml
import random
import numpy as np
import csv

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet
from posthoc_learn.haptic import HapticNet, preprocess

from bite_selection_package.config import spanet_config
from posthoc_learn.config import posthoc_config as config

from PIL import Image, ImageEnhance, ImageFilter

class ConBanDataset:
    """
    Dataset of (Context, ActionDis, ActionCon, Loss, PostHoc, DR Category, DR Loss)
    For hyperparameter tuning and training.
    """

    # Re-import from consolidated
    def __init__(self, consolidated_file):
        try:
            if Path(consolidated_file).exists():
                data = np.load(consolidated_file, allow_pickle=True)
            else:
                data = np.load(config.consolidated_dir / consolidatd_file, allow_pickle=True)

            self.name = Path(consolidatd_file).stem

            self.context = data["context"]
            self.actionDis = data["actionDis"]
            self.actionCon = data["actionCon"]
            self.loss = data["loss"]
            self.posthoc = data["posthoc"]
            self.dr_category = data["dr_category"]
            self.dr_loss = data["dr_loss"]
        except KeyError as err:
            print("Invalid Consolidated File, Missing: {0}".format(err))
            sys.exit(-1)
        except IOError as err:
            print("Cannot read Consolidated File: {0}".format(err))
            sys.exit(-1)

        # Validate
        self._validate()

    # Create dataset from data folder
    def __init__(self, name, visual_model, haptic_model):
        # Check datset dir
        self.name = name
        dataset_dir = config.dataset_dir / name
        if not dataset_dir.exists():
            print("No dataset: {0}".format(dataset_dir))
            sys.exit(-1)

        # Check SPANet
        spanet = SPANet(use_rgb=spanet_config.use_rgb, use_depth=False, use_wall=spanet_config.use_wall)
        checkpoint = config.visual_dir / visual_model
        if not checkpoint.exists():
            print("No visual checkpoint: {0}".format(checkpoint))
            sys.exit(-1)
        spanet.load_state_dict(torch.load(checkpoint)['net'])

        if config.use_cuda:
            spanet = spanet.cuda()

        spanet.eval()

        # Check Haptic
        haptic = HapticNet(config.n_haptic_features, config.n_haptic_categories)
        checkpoint = config.haptic_dir / haptic_model
        if not checkpoint.exists():
            print("No haptic checkpoint: {0}".format(checkpoint))
            sys.exit(-1)
        haptic.load_state_dict(torch.load(checkpoint)['state_dict'])
        haptic_features = []
        def feature_hook(module, input, output):
            haptic_features.clear()
            haptic_features.append(output.cpu().detach().clone().numpy().flatten())
        haptic.linear[3].register_forward_hook(feature_hook)

        if config.use_cuda:
            haptic = haptic.cuda()

        haptic.eval()

        # Loop Through Files
        self.context = []
        self.actionDis = []
        self.actionCon = []
        self.loss = []
        self.posthoc = []
        self.dr_category = []
        self.dr_loss = []

        dr_categories = set()
        actionNums = set()

        for result_file in list(dataset_dir.glob('**/result.yaml')):
            # Validate Files
            image_file = result_file.parent / "image.png"
            haptic_file = result_file.parent / "forces.csv"

            if not image_file.exists():
                print("Warning: cannot find file {0}, skipping".format(image_file))
                continue

            if not haptic_file.exists():
                print("Warning: cannot find file {0}, skipping".format(haptic_file))
                continue

            # Pull from Results File
            with result_file.open() as f:
                result_yaml = yaml.safe_load(f)
                self.actionDis.append(np.array(result_yaml["actionNum"]))
                self.actionCon.append(np.array(result_yaml["action"]))
                self.dr_category.append(np.array(result_yaml["foodName"]))
                dr_categories.add(result_yaml["foodName"])
                actionNums.add(result_yaml["actionNum"])
                if result_yaml["success"]:
                    self.loss.append(0)
                else:
                    self.loss.append(1)

            # Get Context (Visual Features)
            rgb_img = Image.open(image_file)
            if rgb_img.mode != 'RGB':
                rgb_img = rgb_img.convert('RGB')
            rgb_img = ConBanDataset._resize_img(rgb_img, 'RGB')

            transform = transforms.Compose([
                transforms.ToTensor()])

            rgb_img = transform(rgb_img)
            rgb_img = torch.stack([rgb_img])

            if config.use_cuda:
                rgb_img = rgb_img.cuda()

            _, visual_features = spanet(rgb_img, None)

            self.context.append(visual_features.cpu().detach()[0].numpy().flatten())

            # Get Posthoc (Haptic Features)
            haptic_data = np.loadtxt(haptic_file, delimiter=',', skiprows=1)
            haptic_data = preprocess(haptic_data)

            # Crop to 6D, truncate to first frew rows after contact
            haptic_data = haptic_data[:config.n_haptic_dims, :config.n_haptic_rows]

            # Pad to number of rows if necessary
            if haptic_data.shape[1] < config.n_haptic_rows:
                haptic_data = np.pad(haptic_data, ((0, 0), (0, config.n_haptic_rows - haptic_data.shape[1])), 'edge')

            haptic_data = transform(haptic_data).float()

            if config.use_cuda:
                haptic_data = haptic_data.cuda()

            haptic(haptic_data)
            self.posthoc.append(haptic_features[0])

        # Convert lists to np arrays
        assert len(self.context) > 0
        self.context = np.vstack(self.context)
        self.posthoc = np.vstack(self.posthoc)
        self.actionDis = np.array(self.actionDis)
        self.actionCon = np.vstack(self.actionCon)
        self.loss = np.array(self.loss).flatten()
        self.dr_category = np.vstack(self.dr_category)

        # Creating DR Loss Vectors
        lhats = {}
        for cat in self.dr_category:
            lhats[cat[0]] = 0.5 * np.ones(max(actionNums) + 1)
            for action in actionNums:
                action_match = (self.actionDis == action).flatten()
                cat_match = (self.dr_category == cat[0]).flatten()
                truth_arr = np.logical_and(cat_match, action_match)
                losses = self.loss[truth_arr]
                if len(losses) > 0:
                    lhats[cat[0]][action] = np.mean(losses)


        self.dr_loss = np.zeros((len(self.loss), max(actionNums) + 1))
        for i in range(len(self.loss)):
            self.dr_loss[i, :] = lhats[self.dr_category[i][0]]
            self.dr_loss[i, self.actionDis[i]] += (max(actionNums) + 1) * (self.loss[i] - self.dr_loss[i, self.actionDis[i]])


        # Validate
        self.validate()


    # Resize Image for SPANet
    @staticmethod
    def _resize_img(img_org, image_mode='RGB'):
        target_size = spanet_config.img_res
        ratio = float(target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(target_size - new_size[0]) // 2,
                (target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, Image.ANTIALIAS)

        img = Image.new(image_mode, (target_size, target_size))
        img.paste(img_org, pads)
        return img

    # Check that dataset is kosher
    def validate(self):
        print("### Data Validation: " + self.name + " ###")

        # Array Sizes
        data_amt = self.context.shape[0]
        print("Data Amount: {0}".format(data_amt))

        print("Context Dimension: {0}".format(self.context.shape[1]))
        print("Posthoc Dimension: {0}".format(self.posthoc.shape[1]))
        print("Action Dimenstion: {0}".format(self.actionCon.shape[1]))

        actionNum = np.amax(self.actionDis) + 1
        print("Number of Actions: {0}".format(actionNum))

        assert self.posthoc.shape[0] == data_amt
        assert len(self.actionDis) == data_amt
        assert len(self.loss) == data_amt
        assert self.actionCon.shape[0] == data_amt
        assert len(self.dr_category) == data_amt
        assert self.dr_loss.shape == (data_amt, actionNum)

        # DR is Sane
        for i in range(self.dr_loss.shape[0]):
            assert np.amax(self.dr_loss[i, :]) <= actionNum
            assert np.amin(self.dr_loss[i, :]) >= (1.0-actionNum)

            masked = np.copy(self.dr_loss[i, :])
            if np.amax(self.dr_loss[i, :]) > 1.0:
                masked = np.delete(masked, np.argmax(masked))
            elif np.amin(self.dr_loss[i, :]) < 0.0:
                masked = np.delete(masked, np.argmin(masked))

            for num in masked:
                if not (num <= 1.0 or num >= 0.0):
                    print("Error in DR Loss: ")
                    print(self.dr_loss[i, :])
                    print(masked)
                assert num <= 1.0
                assert num >= 0.0


        print("### End Data Validation: " + self.name + " ###")
        print()


    # Export dataset to consolidated file (npz compressed)
    # If file_override = None, save to consolidated_dir / name.npz
    def export(self, file_override = None):
        if file_override is None:
            file_override = config.consolidated_dir / self.name

        np.savez_compressed(file_override, 
            context = self.context,
            actionDis = self.actionDis,
            actionCon = self.actionCon,
            loss = self.loss,
            posthoc = self.posthoc,
            dr_category = self.dr_category,
            dr_loss = self.dr_loss)
