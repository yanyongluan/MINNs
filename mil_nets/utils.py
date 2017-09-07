import numpy as np
from random import shuffle
from keras.utils import np_utils

# save batches, per batch contains instance features of a bag and bag label
def convertToBatch(bags):
    """Convert to batch format.
    Parameters
    -----------------
    bags : list
        A list contains instance features of bags and bag labels.
    Return
    -----------------
    data_set : list
        Convert dataset to batch format(instance features, bag label).
    """
    batch_num = len(bags)
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = np.asarray(bag[0], dtype='float32')
        batch_label = np.asarray(bag[1])
        data_set.append((batch_data, batch_label))
    return data_set

def save_result(filename, run, n_fold, acc):
    """Save experimental result to file.
    Parameters
    -----------------
    filename : string
        Name of file to save experimental result.
    run : int
        Times of K-Fold cross-validation experiments.
    n_fold : int
        Number of K-Fold.
    acc : array (run x n_fold)
        Accuracy array of $run$ times $n_fold$-Fold cross-validation experiments.
    Return
    -----------------
    No return.
    """
    with open(filename, 'w') as f:
        for i in range(run):
            f.write('run='+str(i))
            for j in range(n_fold):
                f.write('  %.3f' % acc[i][j])
            f.write('\n')
        f.write('mean=  %.3f' % np.mean(acc) + '\n')
        f.write('std=   %.3f' % np.std(acc) +' \n')
        f.close()
