import numpy as np
import sys
import time
import random
from random import shuffle
import argparse

from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout

from mil_nets.dataset import load_dataset
from mil_nets.layer import Feature_pooling
from mil_nets.metrics import bag_accuracy
from mil_nets.objectives import bag_loss
from mil_nets.utils import convertToBatch

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train a MI-Net')
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to train on, like musk1 or fox',
                        default=None, type=str)
    parser.add_argument('--pooling', dest='pooling_mode',
                        help='mode of MIL pooling',
                        default='max', type=str)
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=5e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=50, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training MI-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    for ibatch, batch in enumerate(test_set):
        result = model.test_on_batch({'input':batch[0]}, {'fp':batch[1]})
        test_loss[ibatch] = result[0]
        test_acc[ibatch][0] = result[1]
    return np.mean(test_loss), np.mean(test_acc)

def train_eval(model, train_set):
    """Evaluate on training set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training MI-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on traing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    num_train_batch = len(train_set)
    train_loss = np.zeros((num_train_batch, 1), dtype=float)
    train_acc = np.zeros((num_train_batch, 1), dtype=float)
    shuffle(train_set)
    for ibatch, batch in enumerate(train_set):
        result = model.train_on_batch({'input':batch[0]}, {'fp':batch[1]})
        train_loss[ibatch] = result[0]
        train_acc[ibatch][0] = result[1]
    return np.mean(train_loss), np.mean(train_acc)

def MI_Net(dataset):
    """Train and evaluate on MI-Net.
    Parameters
    -----------------
    dataset : dict
        A dictionary contains all dataset information. We split train/test by keys.
    Returns
    -----------------
    test_acc : float
        Testing accuracy of MI-Net.
    """
    # load data and convert type
    train_bags = dataset['train']
    test_bags = dataset['test']

    # convert bag to batch
    train_set = convertToBatch(train_bags)
    test_set = convertToBatch(test_bags)
    dimension = train_set[0][0].shape[1]

    # data: instance feature, n*d, n = number of training instance
    data_input = Input(shape=(dimension,), dtype='float32', name='input')

    # fully-connected
    fc1 = Dense(256, activation='relu', kernel_regularizer=l2(args.weight_decay))(data_input)
    fc2 = Dense(128, activation='relu', kernel_regularizer=l2(args.weight_decay))(fc1)
    fc3 = Dense(64, activation='relu', kernel_regularizer=l2(args.weight_decay))(fc2)

    # dropout
    dropout = Dropout(rate=0.5)(fc3)

    # features pooling
    fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(args.weight_decay), pooling_mode=args.pooling_mode, name='fp')(dropout)

    model = Model(inputs=[data_input], outputs=[fp])
    sgd = SGD(lr=args.init_lr, decay=1e-4, momentum=args.momentum, nesterov=True)
    model.compile(loss=bag_loss, optimizer=sgd, metrics=[bag_accuracy])

    # train model
    t1 = time.time()
    num_batch = len(train_set)
    for epoch in range(args.max_epoch):
        train_loss, train_acc = train_eval(model, train_set)
        test_loss, test_acc = test_eval(model, test_set)
        print 'epoch=', epoch, '  train_loss= {:.3f}'.format(train_loss), '  train_acc= {:.3f}'.format(train_acc), '  test_loss={:.3f}'.format(test_loss), '  test_acc= {:.3f}'.format(test_acc)
    t2 = time.time()
    print 'run time:', (t2-t1) / 60, 'min'
    print 'test_acc={:.3f}'.format(test_acc)

    return test_acc

if __name__ == '__main__':

    args = parse_args()

    print 'Called with args:'
    print args

    # perform five times 10-fold cross-validation experiments
    run = 5
    n_folds = 10
    acc = np.zeros((run, n_folds), dtype=float)
    for irun in range(run):
        dataset = load_dataset(args.dataset, n_folds)
        for ifold in range(n_folds):
            print 'run=', irun, '  fold=', ifold
            acc[irun][ifold] = MI_Net(dataset[ifold])
    print 'MI-Net mean accuracy = ', np.mean(acc)
    print 'std = ', np.std(acc)
