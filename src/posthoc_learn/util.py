import numpy as np
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
import os
import math

# item_list = ['apple/','banana/','blackberry/','cantaloupe/','carrot/','celery/','egg/','grape/','pepper/','strawberry/','tomato/','watermelon/']
categories = ['hard_shell', 'hard', 'medium', 'soft']
item_list = ['banana/', 'carrot/', 'grape/', 'strawberry/', 'broccoli/']
category_dict = {
    'hard':['carrot/', 'broccoli/'],
    'medium':['strawberry/'],
    'soft':['banana/'],
    'hard_shell':['grape/']
}
# category_dict = {
#     'shell':['grape/', 'pepper/','tomato/'],
#     'hard':['apple/', 'carrot/', 'celery/'],
#     'medium':['cantaloupe/','strawberry/','watermelon/'],
#     'soft':['banana/','blackberry/','egg/']}
color_list = ['blue', 'green', 'red', 'black', 'magenta', 'cyan', 'yellow', 'purple']
dim_names= ["fx", "fy", "fz", "tx", "ty", "tz"]#,
            # "dfx", "dfy", "dfz", "dtx", "dty", "dtz",
            # "x", "y","z", "rx", "ry", "rz",
            # "dx", "dy", "dz", "drx", "dry", "drz"]

def create_permutation_for_kfold(size, kfold, filename):
    permutation = np.random.permutation(np.arange(size))
    np.savez(filename, "wb", permutation=permutation)


def get_dims(use_dims):

    # if use_dims is None or len(use_dims) == 0:
    #     return np.arange(24)

    dims = []
    for dim_name in dim_names:
        dims += [dim_names.index(dim_name)]
    print("Use only {}: {}".format(use_dims, dims))
    return dims


def preprocess_robot_data(data, sorted_keys, meanX, stdX, itemize=False, use_dims=[], cut_at_max_of="fz"):

    X = []
    Y = []

    dims = get_dims(use_dims)
    cut_dim_idx = get_dims([cut_at_max_of])

    for category in category_dict.keys():
        for item in category_dict[category]:
            for d in data[item]:
                x = np.copy(d)

                # if (cut_at_max_of is not None and cut_at_max_of is not ""):
                #     idx = np.argmax(d[cut_dim_idx,:])
                #     x = x[:, :idx]

                if len(dims) > 0 and len(dims) != x.shape[0]:
                    x = x[dims, :]

                n_features = x.shape[0]
                x = x.reshape(x.shape[0], 1, x.shape[1])

                X.append(x)
                if itemize:
                    Y.append(Variable(torch.LongTensor([sorted_keys.index(item)]).cuda()))
                else:
                    Y.append(Variable(torch.LongTensor([sorted_keys.index(category)]).cuda()))

    n_features = x.shape[0]
    rawX = [np.copy(x) for x in X]
    X = [(x - np.tile(meanX, x.shape[2]).reshape(n_features,1,x.shape[2])) \
          /np.tile(stdX, x.shape[2]).reshape(n_features,1,x.shape[2]) for x in X]
    X = [Variable(torch.from_numpy(x).float().cuda()) for x in X]



    return X, Y, rawX

def preprocess(data, sorted_keys, itemize, use_dims=[]):

    X = []
    Y = []
    dims = get_dims(use_dims)

    for category in category_dict.keys():
        for item in category_dict[category]:
            for d in data[item]:
                x = np.copy(d)

                if len(dims) > 0 and len(dims) != x.shape[0]:
                    x = x[dims, :]

                n_features = x.shape[0]
                x = x.reshape(x.shape[0], 1, x.shape[1])

                X.append(x)
                if itemize:
                    Y.append(Variable(torch.LongTensor([sorted_keys.index(item)]).cuda()))
                else:
                    Y.append(Variable(torch.LongTensor([sorted_keys.index(category)]).cuda()))
    catX = np.concatenate(X, axis=2)
    meanX = np.mean(catX, axis=2)
    stdX = np.std(catX, axis=2)

    rawX = [np.copy(x) for x in X]

    X = [(x - np.tile(meanX, x.shape[2]).reshape(n_features,1,x.shape[2])) \
          /np.tile(stdX, x.shape[2]).reshape(n_features,1,x.shape[2]) for x in X]
    X = [Variable(torch.from_numpy(x).float().cuda()) for x in X]

    return X, Y, meanX, stdX, rawX

def resize(data, out_dim):
    n_features = data[0].shape[0]
    processed_data = []

    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((n_features, out_dim))])

    for x in data:
        img = x.view(x.size(0), x.size(2)).cpu().data.numpy()[:,:,None]
        out = data_transform(img)
        out = Variable(torch.from_numpy(np.asarray(out)).float().cuda())
        if out.size(0) == out_dim:
            out = out.permute(1,0)
        processed_data += [out]

    return processed_data

def categoryFromOutput(sorted_keys, output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i.cpu().numpy()[0][0]
    return sorted_keys[category_i], category_i

def save_checkpoint(state, is_best, filename, saved_checkpoint_filename=None):
    if saved_checkpoint_filename is None:
        saved_checkpoint_filename = filename

    # if is_best:
    print("Save {}".format(saved_checkpoint_filename))
    torch.save(state, saved_checkpoint_filename)
        # shutil.copyfile(filename, saved_checkpoint_filename)

def get_filename(epoch, model_name, item=False, kfold=None, kth_fold=None, optimizer_name="Adam", prefix=""):

    filename = model_name

    if item:
        filename += "checkpoint_item"
    else:
        filename += "checkpoint_category"

    if kfold is not None and kth_fold is not None:
        filename += "_" + str(kth_fold) + "_of_" + str(kfold) + "fold"

    filename += "_" + optimizer_name

    if epoch is not None:
        filename += "_epoch_" + str(epoch) + ".pth.tar"
    else:
        filename += ".pth.tar"

    import os
    return os.path.join(prefix, filename)

def confusion_matrix_single(model, evaluate, criterion, sorted_keys, testX, testY, filename):

    n_categories = len(sorted_keys)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    # n_confusion = 100

    if n_categories > 4:
        items = []
        key_to_item = dict()
        for category in categories:
            items.extend(category_dict[category])
        for i, key in enumerate(sorted_keys):
            idx = items.index(key)
            key_to_item[i] = idx

    if len(sorted_keys) == 4:
        sorted_keys = [x.title() for x in sorted_keys]
    else:
        sorted_keys = [x[:-1].title() for x in sorted_keys]

    accuracies = []
    predictions = []

    actuals = []
    model.eval()
    _, accuracy, prediction, actual = evaluate(model, criterion, sorted_keys, testX, testY)
    accuracies += [accuracy]
    predictions += [prediction]
    actuals += [actual]

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(len(prediction)):
        if n_categories > 4:
            confusion[key_to_item[prediction[i]]][key_to_item[actual[i]]] += 1
        else:
            confusion[prediction[i]][actual[i]] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    predictions = np.hstack(prediction)
    actuals = np.hstack(actual)
    accuracy = np.sum(actuals == predictions) / float(len(actuals))
    ste = math.sqrt(accuracy * (1-accuracy) / float(len(actuals)))

    f, ax = plt.subplots(1,1, figsize=(7,7))
    cax = ax.imshow(confusion.numpy(), interpolation='nearest', cmap=plt.cm.Greys)

    tick_marks = np.arange(len(sorted_keys))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(sorted_keys, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(sorted_keys)


    # ticklines = ax.get_xticklines() + ax.get_yticklines()
    # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    if len(sorted_keys) > 4:
        ax.set_xticks(np.arange(-.5, len(sorted_keys), 3), minor=True);
        ax.set_yticks(np.arange(-.5, len(sorted_keys), 3), minor=True);
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    else:
        ax.set_xticks(np.arange(-.5, len(sorted_keys), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(sorted_keys), 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    fmt = '.2f'
    thresh = confusion.max() / 2.
    from itertools import product
    for i, j in product(range(confusion.shape[0]), range(confusion.shape[1])):
        ax.text(j, i, format(confusion[i, j], fmt),horizontalalignment="center",color="white" if confusion[i, j] > thresh else "black")
    plt.tight_layout()
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')

    plt.savefig(filename[:-4] + '.pdf', format='pdf', dpi=100)
    print ("Wrote to {}".format(filename[:-4]+".pdf"))
    with open(filename[:-4] +".txt", "w") as f:
        print (accuracy, ste)
        f.write("Accuracy\t{}\tSTE\t{}".format(accuracy, ste))
        print ("Wrote to {}".format(filename[:-4]+".txt"))



def confusion_matrix(models, evaluate, criterion, sorted_keys, testXs, testYs, filename):

    n_categories = len(sorted_keys)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    # n_confusion = 100

    if n_categories > 4:
        items = []
        key_to_item = dict()
        for category in categories:
            items.extend(category_dict[category])
        for i, key in enumerate(sorted_keys):
            idx = items.index(key)
            key_to_item[i] = idx

    if len(sorted_keys) == 4:
        sorted_keys = [x.title() for x in sorted_keys]
    else:
        sorted_keys = [x[:-1].title() for x in sorted_keys]

    accuracies = []
    predictions = []

    actuals = []
    for model, testX, testY in zip(models, testXs, testYs):
        model.eval()
        _, accuracy, prediction, actual = evaluate(model, criterion, sorted_keys, testX, testY)
        accuracies += [accuracy]
        predictions += [prediction]
        actuals += [actual]

        # Go through a bunch of examples and record which are correctly guessed
        for i in range(len(prediction)):
            if n_categories > 4:
                confusion[key_to_item[prediction[i]]][key_to_item[actual[i]]] += 1
            else:
                confusion[prediction[i]][actual[i]] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    predictions = np.hstack(prediction)
    actuals = np.hstack(actual)
    accuracy = np.sum(actuals == predictions) / float(len(actuals))
    ste = math.sqrt(accuracy * (1-accuracy) / float(len(actuals)))

    f, ax = plt.subplots(1,1, figsize=(10,10))
    cax = ax.imshow(confusion.numpy(), interpolation='nearest', cmap=plt.cm.Greys)

    tick_marks = np.arange(len(sorted_keys))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(sorted_keys, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(sorted_keys)


    # ticklines = ax.get_xticklines() + ax.get_yticklines()
    # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    if len(sorted_keys) > 4:
        ax.set_xticks(np.arange(-.5, len(sorted_keys), 3), minor=True);
        ax.set_yticks(np.arange(-.5, len(sorted_keys), 3), minor=True);
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    else:
        ax.set_xticks(np.arange(-.5, len(sorted_keys), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(sorted_keys), 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    fmt = '.2f'
    thresh = confusion.max() / 2.
    from itertools import product
    for i, j in product(range(confusion.shape[0]), range(confusion.shape[1])):
        ax.text(j, i, format(confusion[i, j], fmt),horizontalalignment="center",color="white" if confusion[i, j] > thresh else "black")
    plt.tight_layout()
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')

    plt.savefig(filename[:-4] + '.pdf', format='pdf', dpi=100)
    print ("Wrote to {}".format(filename[:-4]+".pdf"))
    with open(filename[:-4] +".txt", "w") as f:
        print (accuracy, ste)
        f.write("Accuracy\t{}\tSTE\t{}".format(accuracy, ste))
        print ("Wrote to {}".format(filename[:-4]+".txt"))

def load_model(model, optimizer, filename):

    if os.path.isfile(filename):
        print('=> Loading checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> Loaded checkpoint (last epoch: %d)' %
              checkpoint['epoch'])
    else:
        print('=> No checkpoint found at %s' % filename)
    return model

def load_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return optimizer

def get_training_and_test_sets(X, Y, kfold, permutation_filename, indice=None, make=False):

    if not (os.path.exists(permutation_filename)):
        if make:
            create_permutation_for_kfold(len(X), kfold, permutation_filename)
        else:
            raise ValueError("{} not found.".format(permutation_filename))

    permutation = np.load(permutation_filename)['permutation']
    window_size = int(len(X) / kfold)

    testXs = []
    testYs = []
    trainXs = []
    trainYs = []

    for kth_fold in range(kfold):
        indices = permutation[kth_fold*window_size:(kth_fold+1)*window_size]
        testX = [x for i, x in enumerate(X) if i in indices]
        testY = [x for i, x in enumerate(Y) if i in indices]
        train_X = [x for i, x in enumerate(X) if i not in indices]
        train_Y = [x for i, x in enumerate(Y) if i not in indices]
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        testXs += [testX]
        testYs += [testY]

        trainX = []
        trainY = []
        for i in indices:
            trainX += [train_X[i]]
            trainY += [train_Y[i]]
        trainXs += [trainX]
        trainYs += [trainY]

        print("{}-th fold".format(kth_fold))
        print ("Training set size: {}\t Test set size {}".format(len(trainX), len(testX)))


    if indice is None:
        return trainXs, trainYs, testXs, testYs
    else:
        return trainXs[indice], trainYs[indice], testXs[indice], testYs[indice]


def compare_models():
    import glob
    files = glob.glob("*.txt")

    models = []
    accuracies = []
    stes = []
    for f in files:
        model_name = f.split("_")[0]
        with open(f) as data:
            accuracy = data.split("\t")[1]
            ste = data.split("\t")[-1].split(" ")[-1]
            models.append(model_name)
            accuracies.append(accuracy)
            stes.append(ste)

    from matplotlib import plt
    x = np.arange(len(models))
    plt.errorbar(x, accuracies, stes)

def parse_results(rootdir="models/"):
    import os

    results = []
    for dirName, subDir, files in os.walk(rootdir):
        results += [os.path.join(dirName, f) for f in files if f.endswith(".txt")]

    parsed_results = []
    for result in results:
        used_feature = result.split("/")[-2]
        model_name = result.split("/")[-1].split("_")[0]
        if model_name == "LSTM":
            continue
        if "item" in result:
            continue

        with open(result, "r") as r:
            print (result)
            line = r.readline()
            accuracy = float(line.split("\t")[1])
            ste = float(line.split("\t")[-1].split(" ")[-1])

        parsed_results += [(model_name, used_feature, accuracy, ste)]

    return parsed_results

if __name__ == "__main__":
    results = parse_results()

    mlp_results = [x for x in results if x[0] == "MLP"]
    lstm_results = [x for x in results if x[0] == "LSTM"]

    features = [
        ('all_features', 'All'),
        ('forces_and_positions', 'F/P'), # TODO: change this with one with RoC
        ('forces_and_roc', 'Forces'),
        ('torques_and_roc', 'Torques'), # Change this with one with force + torque + RoC
        # ('positions_and_roc', 'Positions'), # Change this with one with force + torque + RoC
        ('z_force_and_roc', 'Z force')
    ]

    colors = ['grey', 'grey', 'red', 'grey','orange']
    mlp_means = []
    mlp_stes = []
    for f in features:
        mlp_means += [x[2] for x in mlp_results if x[1] == f[0]]
        mlp_stes += [x[3] for x in mlp_results if x[1] == f[0]]

    # lstm_means = [x[2] for x in lstm_results]
    # lstm_stes = [x[3] for x in lstm_results]

    series = [mlp_means] #, lstm_means]
    series_labels = ['MLP'] #, 'LSTM']
    series_errs = [mlp_stes] #, lstm_stes]

    plot_labels = [f[1] for f in features]
    from ss_plotting.make_plots import plot_bar_graph
    fig, ax = plot_bar_graph(series, series_labels=series_labels, series_errs=series_errs,
        series_colors=['grey'],
        category_labels=[f[1] for f in features], category_rotation=45, xpadding=0.3, show_plot=False, simplify=False)
    ax.set_ylim([0.70,0.85])
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0], xlim[1]-0.5])
    from ss_plotting import plot_utils
    plot_utils.simplify_axis(ax)
    plt.savefig("Feature_comparison.eps", format='eps', dpi=1000)
    import IPython; IPython.embed()

    # ALl,  3F + RoC, 3F/P + RoC, 3F/T + ROC, Z-force + RoC


# def tapo_plot_confusion_matrix(cm, classes, modality, path, normalize=False, title='Confusion matrix', cmap=pp.cm.Blues):
#     """ This function prints and plots the confusion matrix.
#         Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         cm = cm.astype('int')
#         print('Confusion matrix, without normalization')
#     f, axarr = pp.subplots(1,1)
#     cax = axarr.imshow(cm, interpolation='nearest', cmap=cmap)
#     axarr.set_title(title)
#     cbar = f.colorbar(cax)
#     tick_marks = np.arange(len(classes))
#     axarr.set_xticks(tick_marks)
#     axarr.set_xticklabels(classes, tick_marks, rotation=45)
#     axarr.set_yticks(tick_marks)
#     axarr.set_yticklabels(classes, tick_marks)
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         axarr.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
#     pp.tight_layout()
#     axarr.set_xlabel('True label')
#     axarr.set_ylabel('Predicted label')
#     savepath = path + modality + '_svm.png'
#     f.savefig(savepath)