"""Provides methods for running the `GetAction` and `PublishLoss` ROS services"""

from posthoc_learn.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse
import numpy as np
import rospy
import time, os
import rospkg
from pathlib import Path

rospack = rospkg.RosPack()

from bite_selection_package.config import spanet_config as spanet_config
from posthoc_learn.config import posthoc_config as config

from cv_bridge import CvBridge
from PIL import Image

N_FEATURES = 2049 if spanet_config.n_features==None else spanet_config.n_features+1
PACKAGE_NAME = 'posthoc_learn'
TIME = 0

def _handle_get_action(req, algo, dataset, verbose=True):
    global TIME
    if verbose:
        print('GetAction: called')

    npImg = CvBridge().imgmsg_to_cv2(req.image, desired_encoding='rgb8')
    pilImg = Image.fromarray(npImg)

    # Unflatten visual features.
    features = dataset.get_visual_features(pilImg)
    assert len(features) == features.size, "Multidimensional context"
    assert len(features) == N_FEATURES, "Malformed context"

    # Sample action
    a_t, p_t = algo.choice(TIME, features)

    if verbose:
        print('GetAction: responding with a_t={} and len(p_t)={}'.format(a_t, len(p_t)))

    TIME = TIME + 1
    return GetActionResponse(a_t, p_t.flatten())

def _handle_publish_loss(req, algo, dataset, path, verbose=True):
    if verbose:
        print('PublishLoss: called with a_t={} loss={} len(p_t)={} len(haptic)={}'.format(req.a_t, req.loss, len(req.p_t), len(req.haptic)))
    try:
        # Convert everything to output array
        p_t = np.array(req.p_t)
        npImg = CvBridge().imgmsg_to_cv2(req.image, desired_encoding='rgb8')
        pilImg = Image.fromarray(npImg)
        visual = dataset.get_visual_features(pilImg)
        haptic = np.array(req.haptic)
        loss = float(req.loss)
        a_t = int(req.a_t)

        # Assertions to Ensure Correct Functionality
        assert len(visual) == N_FEATURES, "Malformed Context"
        assert len(visual) == visual.size, "Multidimensional context"

        assert len(haptic) % (config.n_haptic_dims + 1) == 0, "Malformed post hoc"
        haptic = np.reshape(haptic, (-1, config.n_haptic_dims + 1))

        assert a_t < len(p_t), "Malformed action"

        # Process haptic data
        haptic = dataset.get_haptic_features(haptic)

        # Save output result
        file_name = "time_{}.npz".format(time.time())
        file = str(path / file_name)
        print("Saving file: " + file)
        np.savez_compressed(file, 
            context = visual,
            posthoc = haptic,
            loss = loss,
            arm = a_t,
            prob_vec = p_t)

        # Learning
        algo.update(visual, a_t, loss, haptic, p_t[a_t])
    except AssertionError as err:
        print("ERROR: {0}".format(err))
        return PublishLossResponse(success=False)
    return PublishLossResponse(success=True)

def start_get_action(algo, dataset, verbose=True):
    """Starts the `GetAction` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_get_action(req, algo, dataset, verbose=verbose)
    rospy.Service('GetAction', GetAction, handle_wrapper)

def start_publish_loss(algo, dataset, path, verbose=True):
    """Starts the `PublishLoss` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_publish_loss(req, algo, dataset, path, verbose=verbose)
    rospy.Service('PublishLoss', PublishLoss, handle_wrapper)

def create_server(algo, dataset, trial_name, excluded_item=None, server_name=config.server_name, verbose=True):
    """
    Creates the algorithm server with a given algorithm.
    Provides the services `GetAction` and `PublishLoss`.
    """
    rospy.init_node(server_name)
    start_get_action(algo, dataset, verbose=verbose)

    # Pretrain on dataset if requested
    if excluded_item is not None and len(dataset) > 0:
        print("Pretraining... Excluding: " + excluded_item)
        for i in range(len(dataset)):
            context, posthoc, _, a, loss, category = dataset[i]
            if category == excluded_item:
                continue
            algo.update(context, a, loss, posthoc)

    # Handle trial saving folder
    path = Path(rospack.get_path(PACKAGE_NAME)) / "algoserver_results" / trial_name
    if path.is_dir():
        # Train on all NPZ files in folder
        for file in path.glob("*.npz"):
            trial = np.load(file)
            try:
                algo.update(trial["context"], trial["arm"], trial["loss"], trial["posthoc"], trial["prob_vec"][trial["arm"]])
                print("Loaded file: " + str(file))
            except KeyError:
                print("Invalid results file: " + str(file))
                continue
    else:
        # Make Directory
        path.mkdir(parents=True)

    start_publish_loss(algo, dataset, path, verbose=verbose)
