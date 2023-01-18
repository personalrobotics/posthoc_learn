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
    def _init_npz(self, consolidated_file):
        self.haptic = None
        try:
            if Path(consolidated_file).exists():
                data = np.load(consolidated_file, allow_pickle=True)
            elif (config.consolidated_dir / consolidated_file).exists():
                data = np.load(config.consolidated_dir / consolidated_file, allow_pickle=True)
            else:
                data = np.load(config.consolidated_dir / (consolidated_file + ".npz"), allow_pickle=True)

            self.name = Path(consolidated_file).stem

            self.context = data["context"]
            self.actionDis = data["actionDis"]
            self.actionCon = data["actionCon"]
            self.loss = data["loss"]
            self.posthoc = data["posthoc"]
            self.dr_category = data["dr_category"]
            self.dr_loss = data["dr_loss"]
        except KeyError as err:
            print("Invalid Consolidated File, Missing: {0}".format(err))
            raise err
        except IOError as err:
            print("Cannot read Consolidated File: {0}".format(err))
            raise err

        # Validate
        self._validate()

    # Create dataset from data folder
    def __init__(self, name, visual_model=None, haptic_model=None, use_npz=True):
        if use_npz:
            # Try using data from consolidated file first
            try:
                self._init_npz(name)
            except:
                print("Falling back to non-consolidated dataset")
                use_npz = False

        # Check SPANet
        self.spanet = None
        if visual_model is not None:
            self.spanet = SPANet(use_rgb=spanet_config.use_rgb, use_depth=False, use_wall=spanet_config.use_wall)
            checkpoint = config.visual_dir / visual_model
            if not checkpoint.exists():
                print("No visual checkpoint: {0}".format(checkpoint))
                sys.exit(-1)
            self.spanet.load_state_dict(torch.load(checkpoint)['net'])

            if config.use_cuda:
                self.spanet = spanet.cuda()

            self.spanet.eval()
        elif use_npz == False:
            print("Error: need visual model for non-consolidated dataset")
            sys.exit(-1)

        # Check Haptic
        self.haptic = None
        if haptic_model is not None:
            self.haptic = HapticNet(config.n_haptic_dims, config.n_haptic_categories)
            checkpoint = config.haptic_dir / haptic_model
            if not checkpoint.exists():
                print("No haptic checkpoint: {0}".format(checkpoint))
                sys.exit(-1)
            self.haptic.load_state_dict(torch.load(checkpoint)['state_dict'])
            if config.use_cuda:
                self.haptic = self.haptic.cuda()

            self.haptic.eval()
        elif use_npz == False:
            print("Error: need haptic model for non-consolidated dataset")
            sys.exit(-1)

        if use_npz == True:
            # Dataset already built
            print("Using Consolidated Dataset")
            return

        # Loop Through Files
        self.name = name
        self.context = []
        self.actionDis = []
        self.actionCon = []
        self.loss = []
        self.posthoc = []
        self.dr_category = []
        self.dr_loss = []

        # Check datset dir
        dataset_dir = config.dataset_dir / name
        if not dataset_dir.exists():
            print("No existing dataset: {0}".format(dataset_dir))
            print("Warning: Cannot perform pre-training")
            return

        print("Using Raw Dataset")
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
            visual_features = self.get_visual_features(Image.open(image_file))
            self.context.append(visual_features)

            # Get Posthoc (Haptic Features)
            haptic_data = np.loadtxt(haptic_file, delimiter=',', skiprows=1)
            haptic_features = self.get_haptic_features(haptic_data)
            # print("{0}: {1}".format(self.dr_category[-1], np.argmax(haptic_features)))
            self.posthoc.append(haptic_features)

        # Convert lists to np arrays
        assert len(self.context) > 0, "Could not load any valid data"
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
        self._validate()


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
    def _validate(self):
        print("### Data Validation: " + self.name + " ###")

        # Array Sizes
        data_amt = self.context.shape[0]
        print("Data Amount: {0}".format(data_amt))

        print("Context Dimension: {0}".format(self.context.shape[1]))
        print("Posthoc Dimension: {0}".format(self.posthoc.shape[1]))
        print("Action Dimenstion: {0}".format(self.actionCon.shape[1]))

        actionNum = np.amax(self.actionDis) + 1
        print("Number of Actions: {0}".format(actionNum))

        assert self.posthoc.shape[0] == data_amt, "Invalid posthoc"
        assert len(self.actionDis) == data_amt, "Invalid discrete action"
        assert len(self.loss) == data_amt, "Invalid loss"
        assert self.actionCon.shape[0] == data_amt, "Invalid continuous action"
        assert len(self.dr_category) == data_amt, "Invalid DR categories"
        assert self.dr_loss.shape == (data_amt, actionNum), "Invlid DR loss"

        # DR is Sane
        for i in range(self.dr_loss.shape[0]):
            assert np.amax(self.dr_loss[i, :]) <= actionNum, "DR loss too big"
            assert np.amin(self.dr_loss[i, :]) >= (1.0-actionNum), "DR loss too small"

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
                assert num <= 1.0, "DR loss above acceptable range"
                assert num >= 0.0, "DR loss below acceptable range"


        print("### End Data Validation: " + self.name + " ###")
        print()
        self.num_samples = data_amt

    @staticmethod
    def _resize(x, out_dim):
        n_features = x.shape[0]
        data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((n_features, out_dim))])

        img = np.copy(x.view(x.size(0), x.size(2)).data.numpy()[:,:,None])
        out = data_transform(img)
        out = torch.from_numpy(np.asarray(out)).float()
        if out.size(0) == out_dim:
            out = out.permute(1,0)
        return out

    # Run SPANet on visual data
    def get_visual_features(self, rgb_img):
        assert self.spanet is not None, "No SPANet Provided"
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

        # Add bias term
        return np.append(visual_features.cpu().detach()[0].numpy().flatten(), 1.0)

    # Run HapticNet on another bit of haptic data
    def get_haptic_features(self, data):
        assert self.haptic is not None, "No Haptic Model Provided"
        # Make sure it is the dimensionality we need
        assert data.shape[1] == (config.n_haptic_dims + 1), "Malformed haptic data" # Add 1 for Time

        haptic_data = torch.from_numpy(preprocess(data)).float()

        # Resize to length 64 with billinear interpolation
        haptic_data = ConBanDataset._resize(haptic_data, 64)

        if config.use_cuda:
            haptic_data = haptic_data.cuda()

        ret = self.haptic(haptic_data)
        ret = ret.cpu().detach().clone().numpy().flatten()
        # Haptic bias made it much, much worse
        #ret = np.append(ret, 1.0) # Add bias term
        return ret


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

    # List Functionality
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.context[idx, :], self.posthoc[idx, :], self.dr_loss[idx, :], self.actionDis[idx], self.loss[idx], self.dr_category[idx]

    def get_dataset(self, shuffle=True, exclude=None):
        env, _ = self.train_test_split(1.0, suffle, exclude)
        return env

    def train_test_split(self, split, shuffle=True, exclude=None):
        assert (split >= 0.0) and (split <= 1.0), "Malformed Split"

        context = np.copy(self.context)
        posthoc = np.copy(self.posthoc)
        dr_loss = np.copy(self.dr_loss)
        
        if exclude is not None:
            keep = np.where(self.dr_category != exclude)
            context = context[keep, :]
            posthoc = posthoc[keep, :]
            dr_loss = dr_loss[keep, :]

        print("Total Samples:")
        print(context.shape[0])
        if shuffle:
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            context = context[indices, :]
            posthoc = posthoc[indices, :]
            dr_loss = dr_loss[indices, :]

        cut_index = (int)(self.num_samples * split)

        train_context = np.copy(context[:cut_index, :])
        test_context = np.copy(context[cut_index:, :])

        train_posthoc = np.copy(posthoc[:cut_index, :])
        test_posthoc = np.copy(posthoc[cut_index:, :])

        train_dr_loss = np.copy(dr_loss[:cut_index, :])
        test_dr_loss = np.copy(dr_loss[cut_index:, :])

        return (train_context, train_posthoc, train_dr_loss), (test_context, test_posthoc, test_dr_loss)
    
    def sample_with_replacement(self):

        context = np.copy(self.context)
        posthoc = np.copy(self.posthoc)
        dr_loss = np.copy(self.dr_loss)
        print("Total Samples:")
        print(self.num_samples)

        indices = np.arange(self.num_samples)
        # np.random.shuffle(indices)
        indices = np.random.choice(self.num_samples, 115)
        context = context[indices, :]
        posthoc = posthoc[indices, :]
        dr_loss = dr_loss[indices, :]


        return (context, posthoc, dr_loss)
