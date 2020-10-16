# General Configuration

import os
import pathlib
import rospkg
from pathlib import Path
rospack = rospkg.RosPack()
package_dir = Path(rospack.get_path('posthoc_learn'))

use_cuda = True
gpu_id = '0'

# Data configuration
data_name = "initial"
visual_model = "food_spanet_all_rgb_wall_ckpt_best.pth"
haptic_model = "FCNcheckpoint_category_0_of_3fold_Adam.pth"
n_haptic_features = 128
n_haptic_categories = 4
n_haptic_dims = 6
n_haptic_rows = 64

# File Configuration
consolidated_dir = package_dir / "data" / "consolidations"
dataset_dir = package_dir / "data" / "datasets"
visual_dir = package_dir / "checkpoint" / "visual"
haptic_dir = package_dir / "checkpoint" / "haptic"

# Experiment Configuration
train_split = 0.7