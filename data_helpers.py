import os
import scipy.io as sio
import numpy as np
import pickle

from scipy import signal

def load_mat(matfile):
    try:
        mat_content = sio.loadmat(matfile)
        data_struct = mat_content['dataStruct'][0, 0]
        X = data_struct['data']
    except:
        X = None
    return X

def load_spectrogram(matfile):
    X = load_mat(matfile)
    if X is None:
        return None
    fs = 400
    m = []
    for i in range(16):
        x = X[:, i]
        f, t, Sxx = signal.spectrogram(x, fs)
        m.append(Sxx)

    return np.array(m)

def load_data(folder, classes=2):
    files = os.listdir(folder)

    files = [f for f in files if not f.startswith('.')]
    files = np.array([os.path.join(folder, f) for f in files])
    labels = np.array([int(f[-5]) for f in files], dtype=np.float32)

    print "1: ", (labels == 0.0).sum()
    print "0: ", (labels == 1.0).sum()
    
    labels = (np.arange(classes) == labels[:,None]).astype(np.float32)

    return files, labels 

def batch_iter(X, y, batch_size, num_epochs, shuffle=True):
    data_size = X.shape[0]
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_X = X[shuffle_indices]
            shuffled_y = y[shuffle_indices]
        else:
            shuffled_X = X
            shuffled_y = y
        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, data_size)
            batch_files = shuffled_X[start:end]
            batch_labels = shuffled_y[start:end]
  
            X_batch = []
            y_batch = []
            for f, label in zip(batch_files, batch_labels):
                mat_data = load_spectrogram(f)
                if mat_data is not None:
                    X_batch.append(mat_data)
                    y_batch.append(label)

            yield np.array(X_batch), np.array(y_batch)


def batch_iter_balanced(X, y, batch_size, num_epochs, shuffle=True):
    data_size = X.shape[0]
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_X = X[shuffle_indices]
            shuffled_y = y[shuffle_indices]
        else:
            shuffled_X = X
            shuffled_y = y
        for batch_num in range(num_batches_per_epoch):
            indices = np.arange(0, data_size)
            labels = np.argmax(shuffled_y, 1)
            pos_indices = indices[labels == 1]
            neg_indices = indices[labels == 0]
            pos_batch_indices = np.random.choice(pos_indices, batch_size/2)
            neg_batch_indices = np.random.choice(neg_indices, batch_size/2)
            batch_indices = np.concatenate((pos_batch_indices, neg_batch_indices), axis=0)
  
            X_batch = []
            y_batch = []
            for batch_index in batch_indices:
                f =shuffled_X[batch_index]
                label = shuffled_y[batch_index]
                mat_data = load_spectrogram(f)
                if mat_data is not None:
                    X_batch.append(mat_data)
                    y_batch.append(label)

            yield np.array(X_batch), np.array(y_batch)
           
def test():
    X, y = load_data('train_1', classes=2)

    batches = batch_iter(X, y, 4, 1)
    for batch in batches:
        X_batch, y_batch = batch
        print X_batch.shape, y_batch.shape
    return

if __name__ == "__main__":
    test()
