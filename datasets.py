import numpy as np
import pickle

from unbalanced_data import make_unbalanced, make_mnist
from examples import examples as example_embeddings

def get_new_data(dataset, n_rare, prop_rare):
    data = np.load(f"./embeddings/{dataset}.npz")
    examples = pickle.load(open('new_examples.pkl','rb'))[dataset]
    classes = np.array(list(examples.keys()))
    X = data['embeddings']
    y = data['target']
    print(X.shape[0], y.shape[0], y.sum())
    index = np.arange(y.shape[0])
    class_1 = np.random.permutation(index[y == 0])
    class_2 = np.random.permutation(index[y == 1])
    class_1, class_2 = (class_1, class_2) if class_1.shape[0] >= class_2.shape[0] else (class_2, class_1)
    if prop_rare is not None:
        n_rare = int(prop_rare * class_1.shape[0])
    print(f'class 1: {class_1.shape[0]}, class 2: {class_2.shape[0]}, n_rare: {n_rare}')
    mintest = np.min(np.array([int((y == i).sum() * 0.1) for i in classes]))
    n_test = np.max(np.array([200, mintest]))
    test_index = np.random.permutation(np.concatenate([class_1[:n_test], class_2[:n_test]]))
    X_test = X[test_index, ...]
    y_test = y[test_index, ...]
    train_index = np.random.permutation(np.concatenate([class_1[n_test:], class_2[n_test:n_test + n_rare]]))
    X_train = X[train_index, ...]
    y_train = y[train_index, ...]
    print(n_rare, X_train.shape[0], X_test.shape[0], train_index.shape[0])
    return X_train, y_train, X_test, y_test, examples, classes

def get_data(dataset, n_rare=100, prop_rare=None):
    '''
    Make simulated datasets by either subsampling mnist or using sklearn's blobs dataset

    Mnist is passed with the number of epochs of training. 
    E.g. mnist-2 trains the embedding for two epochs, mnist-0 uses a random embedding, etc.

    Bass, bat and bank were from the first version of this paper so have more manual code.
    The rest of the examples all share the get_new_data() method.
    '''
    examples = None
    classes = None
    if dataset == "sklearn":
        py_true = np.array([1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        py_true = py_true / py_true.sum()
        X_train, y_train, py_true, X_test, y_test, mask = make_unbalanced(4, 10, 5e-5, n_samples=2_000_000, py=py_true)
        return X_train, y_train, X_test, y_test, examples, classes
    elif "mnist" in dataset:
        epochs = float(dataset.split("-")[-1])
        X_train, y_train, X_test, y_test = make_mnist(epochs=epochs, device="cuda", verbose=False)
        return X_train, y_train, X_test, y_test, examples, classes
    elif dataset in example_embeddings.keys():
        return get_new_data(dataset, n_rare, prop_rare)
    elif "bass" in dataset:
        data = np.load("./bass.npz")
        example_dic = pickle.load(open("./bass_examples.pkl","rb"))
        examples = {}
        for i, j in zip(['guitar', 'fish', 'bast'], [1, 0, 2]):
            examples[j] = example_dic[i]
        classes = np.arange(3)
        X = data['embeddings']
        y = data['target']
        print(X.shape[0], y.shape[0], y.sum())
        index = np.arange(y.shape[0])
        class_2 = np.random.permutation(index[np.logical_not(y)])
        class_1 = np.random.permutation(index[y])
        print(f'class 1: {class_1.shape[0]}, class 2: {class_2.shape[0]}')
        n_test = 200
        test_index = np.random.permutation(np.concatenate([class_1[:n_test], class_2[:n_test]]))
        X_test = X[test_index, ...]
        y_test = y[test_index, ...]
        train_index = np.random.permutation(np.concatenate([class_1[n_test:], class_2[n_test:n_test + n_rare]]))
        X_train = X[train_index, ...]
        y_train = y[train_index, ...]
        print(n_rare, X_train.shape[0], X_test.shape[0], train_index.shape[0])
        return X_train, y_train, X_test, y_test, examples, classes
    elif "bank" in dataset:
        data = np.load("./bank.npz")
        example_dic = pickle.load(open("./bank_examples.pkl","rb"))
        examples = {}
        for i, j in zip(['river', 'finance', 'set'], [0, 1, 2]):
            examples[j] = example_dic[i]
        classes = np.arange(3)
        X = data['embeddings']
        y = data['target']
        print(X.shape[0], y.shape[0], y.sum())
        index = np.arange(y.shape[0])
        class_2 = np.random.permutation(index[np.logical_not(y)])
        class_1 = np.random.permutation(index[y])
        print(f'class 1: {class_1.shape[0]}, class 2: {class_2.shape[0]}')
        n_test = 50
        test_index = np.random.permutation(np.concatenate([class_1[:n_test], class_2[:n_test]]))
        X_test = X[test_index, ...]
        y_test = y[test_index, ...]
        train_index = np.random.permutation(np.concatenate([class_1[n_test:], class_2[n_test:n_test + n_rare]]))
        X_train = X[train_index, ...]
        y_train = y[train_index, ...]
        print(n_rare, (y_train == 1).sum(),(y_train == 0).sum(), X_train.shape[0], X_test.shape[0], train_index.shape[0])
        return X_train, y_train, X_test, y_test, examples, classes
    elif "bat" in dataset:
        data = np.load("./bat.npz")
        example_dic = pickle.load(open("./bat_examples.pkl","rb"))
        examples = {}
        for i, j in zip(['stick', 'creature', 'action'], [0, 1, 2]):
            examples[j] = example_dic[i]
        classes = np.arange(3)
        X = data['embeddings']
        y = data['target']
        print(X.shape[0], y.shape[0], y.sum())
        index = np.arange(y.shape[0])
        class_2 = np.random.permutation(index[np.logical_not(y)])
        class_1 = np.random.permutation(index[y])
        print(f'class 1: {class_1.shape[0]}, class 2: {class_2.shape[0]}')
        n_test = 25
        test_index = np.random.permutation(np.concatenate([class_1[:n_test], class_2[:n_test]]))
        X_test = X[test_index, ...]
        y_test = y[test_index, ...]
        train_index = np.random.permutation(np.concatenate([class_1[n_test:n_test + n_rare], class_2[n_test:]]))
        X_train = X[train_index, ...]
        y_train = y[train_index, ...]
        print(n_rare, (y_train == 1).sum(), (y_train == 0).sum(), X_train.shape[0], X_test.shape[0], train_index.shape[0])
        #exit()
        return X_train, y_train, X_test, y_test, examples, classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

