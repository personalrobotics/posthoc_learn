from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import shutil
import math
from torchvision import transforms
from util import *

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

####### Data Preprocessing ################

# Crop Data

sampleNum = 50 # 50 for normal, 10 for hummus
sampleNum_ = 10
crop_flag = 0
prev_crop_flag = 0
phase_change = 0
minDataNum = 10 # 20 for normal, 10 for hummus
lookBehind_thr = 0.1 # 0.2 for force, 0.0005 for torque
lookAhead_thr = 0.2 # 0.2 for force, 0.0005 for torque
mocap_delay = 3 # 21.3ms, which is 2.556 data points for 120hz
sampleNum_sum = 0

lookAhead_arg = 4
lookBehind_arg = 20
forceThr_arg = 0.1

def crop(time, x_force, y_force, z_force, x_torque, y_torque, z_torque, start_idx = 0, attempt = 1):
    global crop_flag, prev_crop_flag, lookAhead, lookBehind, lookBehind_thr, lookAhead_thr, idx_init, idx_end

    z_force_cropped_t = []
    z_force_idx = []
    x_force_cropped = []
    y_force_cropped = []
    z_force_cropped = []
    xy_force_list = []
    x_torque_cropped = []
    y_torque_cropped = []
    z_torque_cropped = []
    xy_torque_list = []
    xy_torque_cropped_t = []
    # print len(z_force)
    z_force = np.abs(z_force)
    for x, y in zip(x_force, y_force):
        xy_force = math.sqrt(x*x+y*y)
        xy_force_list.append(xy_force)

    for x_t, y_t in zip(x_torque, y_torque):
        xy_torque = math.sqrt(x_t*x_t+y_t*y_t)
        xy_torque_list.append(xy_torque)
    
    maxForce = max(z_force)
    minForce = min(z_force)
    forceThr = forceThr_arg * maxForce
    print len(z_force)

    while not len(z_force_cropped_t)>minDataNum:
        z_time = []
        z_force_cropped_t = []
        z_force_idx = []
        for idx,z in enumerate(z_force):
            # skip to correct range
            if idx < start_idx:
              continue

            prev_crop_flag = crop_flag
            if float(z)>forceThr: # if z_force is bigger than forceThr
                crop_flag = 1
                if prev_crop_flag != crop_flag and crop_flag == 1: # take the value one step before the beginning of the crop
                    z_force_cropped_t.append(z_force[idx-1])
                    z_time.append(time[idx-1])
                    z_force_idx.append(idx-1)
                z_force_cropped_t.append(z)
                z_time.append(time[idx])
            elif (idx+15>=len(z_force) or float(z_force[idx+15]) < forceThr) and (idx+30>=len(z_force) or float(z_force[idx+30]) < forceThr):
                crop_flag = 0
                if prev_crop_flag != crop_flag and crop_flag == 0:
                    z_force_cropped_t.append(z)
                    z_time.append(time[idx])
                    z_force_idx.append(idx)
            if prev_crop_flag != crop_flag and crop_flag == 0:
                break
        if len(z_force_cropped_t)==0:
            break
        forceThr = forceThr+0.01

    print z_force_idx
    if len(z_force_idx) < 1:
      print("cropping failed")
      return False, -1, -1

    idx_init = int(z_force_idx[0])
    idx_end = int(len(z_force)-1)

    z_force_truncated = z_force[idx_init: idx_end]
    if max(z_force_truncated) - min(z_force_truncated) < (maxForce - minForce) * 0.5: # it was just noise after all
      crop(subject_name, food_name, trial, time, x_force, y_force, z_force, x_torque, y_torque, z_torque, start_idx = idx_end+10, attempt = attempt)
      return False, -1, -1

    lookBehind = lookBehind_arg

    lookAhead_thr = 0.4
    lookAhead = lookAhead_arg

    # cropping time
    z_init_time = float(time[idx_init])
    z_time_zero = [float(t)-z_init_time for t in time[idx_init:idx_end+1]]

    # cropping force
    x_force_cropped = x_force[idx_init:idx_end+1]
    y_force_cropped = y_force[idx_init:idx_end+1]    
    z_force_cropped = z_force[idx_init:idx_end+1]

    # cropping torque
    x_torque_cropped = x_torque[idx_init:idx_end+1]
    y_torque_cropped = y_torque[idx_init:idx_end+1]
    z_torque_cropped = z_torque[idx_init:idx_end+1]

    return True, idx_init, idx_end+1

def preprocess(data):
    # data:  time fx fy fz tx ty tz
    # output: fx fy fz tx ty tz
    data1 = np.array(data)
    data = np.transpose(data1)

    sz = data.shape[1]
    done, crop1, crop2 = crop(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
    if done and crop1 >= 0 and crop2 <= sz:
        data = data[:, crop1:crop2]
    time = data[0]
    data = data[1:, :]
    vel = np.diff(data[0:6,:], axis=1)
    vel = np.concatenate([vel, vel[:,-1].reshape(-1,1)], axis=1)
    data = np.concatenate([data[0:6,:], vel, data[6:]], axis=0)

    return data

####### End Data Preprocessing ################

class HapticNet(nn.Module):
    def __init__(self, input_dim, num_categories, dropout=0.1):
        super(HapticNet, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(6*64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_categories))

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = dropout

    def forward(self, input):
        output = self.linear(input.view(-1,6*64))
        if self.dropout > 0 and self.training:
            output = F.dropout(output, training=self.training, p=self.dropout)
        output = self.softmax(output)
        return output


def train(model, category, training_sequence):
    optimizer.zero_grad()
    out = model(training_sequence)
    # category = category.repeat(training_sequence.size(2))
    # loss = criterion(out.view(out.size(1), out.size(2)), category)
    loss = criterion(out, category)
    loss.backward()
    global iteration
    # print(category)
    # print(loss)
    # print(training_sequence.shape)
    iteration += 1
    if iteration % 100 == 0:
        print('Iter{} loss: {}'.format(
            iteration, loss.item() / training_sequence.size(2)))
        # guess, _ = categoryFromOutput(out[0, -1, :])
        guess, _ = categoryFromOutput(sorted_keys, out)
        print("[Training]\tGuess: {}".format(guess),
              "\t\tActual: %10s" % (
                  sorted_keys[category.item()]),
              "\t  {}".format("<<< Correct" if guess == sorted_keys[category.item()] else "<<< x"))

    optimizer.step()


def evaluate(model, testX, testY):

    model.eval()
    print("Evaluate")
    total_test_loss = 0
    hit = 0
    actual = []
    prediction = []
    for input, category in zip(testX, testY):
        out = model(input)
        loss = criterion(out, category)
        # print(category)
        total_test_loss += loss.item() / category.size(0)

        guess, idx = categoryFromOutput(sorted_keys, out)
        actual_idx = category.item()

        # if actual_idx == 0 and idx != 0:
        #     import IPython; IPython.embed()
        actual += [actual_idx]
        prediction += [idx]

        if actual_idx == idx:
            hit += 1

    print('Total loss:\t', total_test_loss / len(testY))
    print('Total accuracy:\t', float(hit) / float(len(testY)))
    test_accuracy = float(hit) / float(len(testY))

    return total_test_loss, test_accuracy, prediction, actual


def train_epoch(model, epoch, item=False, kfold=None, kth_fold=None):

    global training_accuracies
    global test_accuracies
    global best_loss

    model.train()
    for input, target in zip(trainX, trainY):
        train(model, target, input)

    print("Training set evaluation")
    _, training_accuracy, _, _ = evaluate(model, trainX, trainY)
    training_accuracies += [training_accuracy]

    print("Test set evaluation")
    total_test_loss, test_accuracy, _, _ = evaluate(model, testX, testY)
    test_accuracies += [test_accuracy]

    if total_test_loss < best_loss:
        is_best = True
        best_loss = total_test_loss
    else:
        is_best = False

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best, saved_checkpoint_filename)




if __name__ == "__main__":

    from data_process import load_all
    from util import *
    import argparse

    import pickle
    data = pickle.load(open("all_clean_ethan.pkl"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="1")
    parser.add_argument("--item", action="store_true")
    parser.add_argument("--layer", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kfold", type=int, default=3)
    parser.add_argument("--kth-fold", type=int, default=0)
    parser.add_argument("--use_dim", nargs='*')

    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--predictive-accuracy", action="store_true")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--use-all", action="store_true")
    parser.add_argument("--modelpath-prefix", type=str, required=True)

    args = parser.parse_args()


    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    global sorted_keys
    global saved_checkpoint_filename
    global ignore
    global ignore_dim
    global ignore_dim_idx
    global kfold
    global optimizer_name
    global n_features
    global kth_fold
    global iteration
    iteration = 0

    sorted_keys = []
    layer = args.layer
    kfold = args.kfold
    use_dim = args.use_dim

    optimizer_name = args.optimizer

    if args.item:
        for category in categories:
            sorted_keys.extend(category_dict[category])
    else:
        sorted_keys = categories

    if not args.use_all:
        saved_checkpoint_filename = get_filename(None, model_name="FCN", prefix=args.modelpath_prefix,
            item=args.item, optimizer_name=optimizer_name,
            kfold=args.kfold,
            kth_fold=args.kth_fold)
    else:
        saved_checkpoint_filename = get_filename(None, model_name="FCN", prefix=args.modelpath_prefix,
            item=args.item, optimizer_name=optimizer_name,
            kfold=None,
            kth_fold=None)
        kth_fold = None

    print(sorted_keys)
    print ("Checkpoint ", saved_checkpoint_filename)

    X = []
    Y = []

    X, Y, _, _, _= preprocess(data, sorted_keys, args.item, use_dim)
    X = resize(X, 64)

    n_features = X[0].size(0)
    print (n_features)

    X = [x.contiguous().view(1,1,n_features,64) for x in X]


    model = HapticNet(n_features, len(sorted_keys))
    # if use_cuda:
    model.cuda()

    criterion = nn.NLLLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


    if os.path.isfile(saved_checkpoint_filename):
        print('=> Loading checkpoint %s' % saved_checkpoint_filename)
        checkpoint = torch.load(saved_checkpoint_filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> Loaded checkpoint (last epoch: %d)' %
              checkpoint['epoch'])

    else:
        print('=> No checkpoint found at %s' % saved_checkpoint_filename)

    if not args.use_all:

        permutation_filename = "permutation_{}.npz".format(kfold)
        if not (os.path.exists(permutation_filename)) and (not args.predictive_accuracy and not args.confusion):
            if args.kth_fold == 0:
                create_permutation_for_kfold(len(X), 3,permutation_filename)
            else:
                raise ValueError("{} not found.".format(permutation_filename))

        permutation = np.load(permutation_filename)['permutation']
        window_size = int(len(X) / kfold)

        kth_fold = args.kth_fold

        indices = permutation[kth_fold*window_size:(kth_fold+1)*window_size]

        testX = [x for i, x in enumerate(X) if i in indices]
        testY = [x for i, x in enumerate(Y) if i in indices]
        train_X = [x for i, x in enumerate(X) if i not in indices]
        train_Y = [x for i, x in enumerate(Y) if i not in indices]
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        trainX = []
        trainY = []
        for i in indices:
            trainX += [train_X[i]]
            trainY += [train_Y[i]]
        print ("Training set size: {}".format(len(trainX)))
        print("{}-th fold".format(kth_fold))
    else:
        train_X = X
        train_Y = Y
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        trainX = []
        trainY = []
        for i in indices:
            trainX += [train_X[i]]
            trainY += [train_Y[i]]
        testY = trainY
        testX = trainX


    global training_accuracies
    global test_accuracies
    training_accuracies = []
    test_accuracies = []

    global best_loss
    best_loss = float("inf")
    global best_accuracy
    best_accuracy = 0

    report = saved_checkpoint_filename[:-8]

    if args.confusion:
        confusion_matrix(model, evaluate, sorted_keys, testX, testY, report + ".png")
        exit()

    for epoch in range(100):
        train_epoch(model, epoch, args.item, kth_fold)
