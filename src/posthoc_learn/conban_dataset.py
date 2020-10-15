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
    	spanet = SPANet(use_rgb=spanet_config.use_rgb, use_depth=False, use_wall=config.use_wall)
    	checkpoint = config.visual_dir / visual_model
    	if not checkpoint.exists():
    		print("No visual checkpoint: {0}".format(checkpoint))
    		sys.exit(-1)
    	spanet.load_state_dict(torch.load(checkpoint)['net'])

    	if config.use_cuda:
        	spanet = spanet.cuda()

    	# Check Haptic
    	haptic = HapticNet(config.n_haptic_features, config.n_haptic_categories)
    	checkpoint = config.haptic_dir / haptic_model
    	if not checkpoint.exists():
    		print("No haptic checkpoint: {0}".format(checkpoint))
    		sys.exit(-1)
    	haptic.load_state_dict(torch.load(checkpoint)['state_dict'])
    	haptic_features = None
    	def feature_hook(module, input, output):
    		haptic_features = output.cpu().detach().clone().numpy().flatten()
    	haptic.linear[3].register_forward_hook(feature_hook)

    	if config.use_cuda:
        	haptic = haptic.cuda()

    	# Loop Through Files
    	self.context = []
		self.actionDis = []
		self.actionCon = []
		self.loss = []
		self.posthoc = []
		self.dr_category = []
		self.dr_loss = []

		for result_file in list(dataset_dir.glob('**/result.yaml')):
			# Validate Files
			image_file = result_file / "../image.png"
			haptic_file = result_file / "../forces.csv"

			if not image_file.exists():
				print("Warning: cannot find file {0}, skipping".format(image_file))
				continue

			if not haptic_file.exists():
				print("Warning: cannot find file {0}, skipping".format(haptic_file))
				continue

			# Pull from Results File
			with result_file.open() as f:
				result_yaml = yaml.load(f)
				self.actionDis.append(result_yaml["actionNum"])
				self.actionCon.append(result_yaml["action"])
				self.dr_category.append(result_yaml["foodName"])
				if result_yaml["success"]:
					self.loss.append(0)
				else:
					self.loss.append(1)

			# Get Context (Visual Features)
			rgb_img = Image.open(image_file)
            if rgb_img.mode != 'RGB':
                rgb_img = rgb_img.convert('RGB')
            rgb_img = _resize_img(rgb_img, 'RGB')

            transform = transforms.Compose([
                transforms.ToTensor()])

            rgb_img = transform(rgb_img)

            if config.use_cuda:
            	rgb_img = rgb_img.cuda()

            _, visual_features = spanet(rgb_img, None)

            self.context.append(visual_features.cpu().detach()[0].numpy().flatten())

            # Get Posthoc (Haptic Features)
            haptic_data = np.loadtxt(haptic_file, delimiter=',', skiprows=1)
            haptic_data = preprocess(haptic_data)
            haptic_data = transform(haptic_data.flatten()) # To Flat Tensor

            if config.use_cuda:
            	haptic_data = haptic_data.cuda()

            haptic(haptic_data)
            self.posthoc.append(haptic_features.cpu().detach()[0].numpy().flatten())

        # Convert lists to np arrays

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
    	pass

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
