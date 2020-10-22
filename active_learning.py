from tqdm import tqdm
import warnings
import pickle
import argparse

import numpy as np
import pandas as pd
import sklearn
import multiprocessing as mp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize_scalar
from scipy.stats import entropy
from scipy.special import logsumexp

from matplotlib import pyplot as plt

from unbalanced_data import emp_prob
from plotting import plotperf
from datasets import get_data

def imbalance(y, y_emp, mask=None):
    p = y / y.sum()
    p_emp = y_emp / y_emp.sum() 
    q = np.ones_like(y) / y.shape[0]
    return 1 - entropy(p, q) / entropy(y_emp, q)
    
def softmax(x):
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def kl(p, q):
    if p == 0:
        return np.log(1 / (1 - q))
    else:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def U(a, mu, n=10000):
    u_pot = np.linspace(0, 1 - 1 / n, n)
    const = kl(mu, u_pot) < a
    if const.sum() == 0:
        return 1.0
    u = (u_pot[const]).max()
    return u

def L(a, mu, n=10000):
    u_pot = np.linspace(0, 1 - 1 / n, n)
    const = kl(mu, u_pot) < a
    if const.sum() == 0:
        return 0.0
    u = (u_pot[const]).min()
    return u

def chernoff(z, delta, n=None):
    n = z.shape[0] if n is None else n
    phat = z.mean(axis=0)
    return L(np.log(n) * delta, phat), U(np.log(n) * delta, phat)

def emp_bernstein(z, delta, n=None):
    n = z.shape[0] if n is None else n
    phat = z.mean(axis=0)
    vhat = z.var(axis=0, ddof=1)
    eps = np.sqrt(2 * vhat * np.log(2 / delta) / n) + 7*np.log(2. / delta) / (3 * (n-1))
    return [phat - eps, phat + eps]

class ActiveLearner:
    def __init__(self, x_pool, x_test, y_test, classes=None, mask=None, label_function=None, y_train=None, **kwargs):
        if label_function is None:
            if y_train is not None:
                self.label_function = lambda index: y_train[index, ...]
            else:
                raise ValueError("Must supply y_train if no label function is provided")
        else:
            self.label_function = label_function
        self.x_pool = x_pool
        self.x_pool_unlabeled = np.arange(self.x_pool.shape[0])
        self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
        self.y_train = None
        self.x_train = None
        self.x_test = x_test
        self.y_test = y_test
        self.model_trained = False
        self.classes = classes if classes is not None else np.unique(y_train)
        self.imbalance = lambda p: imbalance(p, np.array([(y_train == i).mean() for i in self.classes]), 
                                             mask)
        self.scores = np.ones(self.x_pool.shape[0]) * np.inf
        self.perf = pd.DataFrame({"queries":[], "imbalance":[], 
                                  "micro_f1":[], "macro_f1":[],
                                  "weighted":[], "accuracy":[],
                                  "class_accuracy":[],
                                  "class_coverage":[]}
                                )
    
    def update_model(self, reset=True):
        if reset:
            self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
        self.model.fit(self.x_train, self.y_train)
        
    def report_scores(self, ave_vals="all"):
        if ave_vals == "all":
            ave_vals = ['micro', 'macro', None]
        elif not isinstance(ave_vals, list):
            ave_vals = [ave_vals]
        y_pred = self.predict(self.x_test) 
        for ave in ave_vals:
            if ave is None:
                warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
                score = f1_score(self.y_test, y_pred, average=ave)
                print("all   "+" ".join([f'{i}:{s:1.2f}' for i, s, in enumerate(score)]))
            else:
                print(ave, f1_score(self.y_test, y_pred, average=ave))
    
    def select_points(self, n=1):
        if n > self.x_pool_unlabeled.shape[0]:
            print(f"Less than {n} unlabeled examples. Return last indices...")
            x, y = self.x_pool[self.x_pool_unlabeled, ...], self.label_function(self.x_pool_unlabeled)
            self.x_pool_unlabeled = np.array([])
            return x,y
        return self._select_points(n)
    
    def _select_points(self, n=1):
        raise NotImplementedError()
        
    def select_and_train(self, batch_size=1):
        x, y = self.select_points(batch_size)
        self.model_trained = True
        if self.y_train is None:
            self.y_train = y
            self.x_train = x
        else:
            self.y_train = np.concatenate([self.y_train, y], axis=0)
            self.x_train = np.concatenate([self.x_train, x], axis=0)
        try:
            self.update_model()
            self.predict = self.model.predict
            self.update_performance()
        except ValueError as err:
            self.model_trained = False
            self.predict = lambda x: np.ones(x.shape[0]) * y[0]
            self.update_performance()
            warnings.warn(f"Unable to train this round - not enough diversity. "+
            f"N samples: {self.y_train.shape[0]}, p_y: {emp_prob(self.y_train)}")
            print(err)
        
    def update_performance(self):
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
        y_pred = self.predict(self.x_test)
        y_train_emp = np.array([(self.y_train == i).sum() for i in self.classes])
        py_train = y_train_emp / y_train_emp.sum()

        y_test = self.y_test
        self.perf = self.perf.append({"queries":self.y_train.shape[0], 
                                      "imbalance":self.imbalance(py_train), 
                                      "micro_f1":f1_score(self.y_test, y_pred, average="micro"), 
                                      "macro_f1":f1_score(self.y_test, y_pred, average="macro"),
                                      "weighted":f1_score(self.y_test, y_pred, average="weighted"),
                                      "accuracy":(y_pred == self.y_test).mean(),
                                      "class_accuracy":[(y_pred[y_test == i,...] == y_test[y_test == i, ...]).mean() 
                                                        for i in np.unique(y_test)],
                                      "class_coverage":np.mean([np.isin(i, self.y_train) for i in np.unique(self.y_test)])},
                                     ignore_index=True)

class RandomSearch(ActiveLearner):
    def _select_points(self, n=1):
        full_idx = np.random.permutation(self.x_pool_unlabeled)
        i = full_idx[0:n]
        self.x_pool_unlabeled = full_idx[n:]
        return self.x_pool[i, ...], self.label_function(i)

class ScoreLearner(ActiveLearner):
    def update_score(self):
        raise NotImplementedError()
    
    def _select_points(self, n=1):
        self.update_score()
        idx = np.argsort(self.scores[self.x_pool_unlabeled])[0:n]
        select = self.x_pool_unlabeled[idx]
        self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, select)
        return self.x_pool[select, ...], self.label_function(select)

class EntropySearch(ScoreLearner):
    def update_score(self):
        try:
            p = self.model.predict_proba(self.x_pool[self.x_pool_unlabeled, ...])
            self.scores[self.x_pool_unlabeled] = (p * np.log(p)).sum(axis=1) # negative entropy
        except sklearn.exceptions.NotFittedError:
            self.scores[self.x_pool_unlabeled] = np.random.rand(self.x_pool_unlabeled.shape[0]) * 10 - 1000 # random scores

class LeastConfidence(ScoreLearner):
    def update_score(self):
        try:
            p = self.model.predict_proba(self.x_pool[self.x_pool_unlabeled, ...])
            self.scores[self.x_pool_unlabeled] = -(1 - p.max(axis=1))
        except sklearn.exceptions.NotFittedError:
            self.scores[self.x_pool_unlabeled] = np.random.rand(self.x_pool_unlabeled.shape[0]) * 10 - 1000 # random scores

class LogisticUCB(ScoreLearner):
    def update_score(self):
        try:
            offset = np.inf
            logits = self.model.predict_log_proba(self.x_pool[self.x_pool_unlabeled, ...])
            p = np.zeros((logits.shape[0], self.classes.shape[0])) - offset
            for i in range(logits.shape[1]):
                p[:,i] = logits[:,i]
            exps = np.exp(p - p.max())
            p = exps / exps.sum(axis=1)[:, None]
            p_var = p * (1-p)
            ucb = p + 2 * np.sqrt(p_var)
            cls, counts = np.unique(self.y_train, return_counts=True)
            class_counts = {c: n for c,n in zip(cls, counts)}
            c = np.array([class_counts.get(c, 0) for c in self.classes])
            util = 1/(np.sqrt(c)+0.01)
            self.scores[self.x_pool_unlabeled] = -(ucb * util).sum(axis=1)
        except sklearn.exceptions.NotFittedError:
            self.scores[self.x_pool_unlabeled] = np.random.rand(self.x_pool_unlabeled.shape[0]) * 10 - 1000 # random scores

class GuidedLearner(ActiveLearner):
    def _select_random(self, n=1):
        full_idx = np.random.permutation(self.x_pool_unlabeled)
        i = full_idx[0:n]
        self.x_pool_unlabeled = full_idx[n:]
        return self.x_pool[i, ...], self.label_function(i)

    def _select_points(self, n=1):
        # we use the true labels to select optimally - so unfair comparision with the rest of the methods
        true_labels = self.label_function(self.x_pool_unlabeled)
        classes = self.classes.copy()
        classes = np.random.permutation(classes)
        batch_size = n // classes.shape[0]
        return_idx = np.zeros((0,),dtype="int")
        # fill the first K-1 classes
        for i in classes[:-1]:
            indices = np.arange(true_labels.shape[0])[true_labels == i]
            idx = np.random.permutation(indices)[0:batch_size]
            return_idx = np.concatenate([return_idx, self.x_pool_unlabeled[idx]])
        # get the rest of the batch
        indices = np.arange(true_labels.shape[0])[true_labels == classes[-1]]
        idx = np.random.permutation(indices)[0:(n-return_idx.shape[0])]
        return_idx = np.concatenate([return_idx, self.x_pool_unlabeled[idx]])
        self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, return_idx)
        if return_idx.shape[0] != n:
            # if the class doesn't exist in the data, select random indices
            random_idx = np.random.permutation(self.x_pool_unlabeled)[0:(n-return_idx.shape[0])]
            return_idx = np.concatenate([return_idx, random_idx])
            self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, return_idx)
        return self.x_pool[return_idx, ...], self.label_function(return_idx)

class ImportanceLearner(ActiveLearner):
    def __init__(self, x_pool, x_test, y_test, exemplar_examples=None, classes=None, mask=None, label_function=None, y_train=None, 
                 random_example=True, min_prob=1e-6, rv_bounds=None, thompson=False, epsilon=0.2):
        classes = classes if classes is not None else np.unique(y_train)
        if exemplar_examples is None:
            exemplar_examples = {}
            for i in classes:
                if random_example:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    idx = indices[np.random.randint(indices.shape[0])]
                else:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    x_ave = np.mean(x_pool[indices, ...], axis=0)
                    dist = euclidean_distances(x_pool[indices, ...], x_ave[None, :])
                    sorted_idx = np.argsort(dist)
                    idx = indices[sorted_idx[0]][0]
                exemplar_examples[i] = x_pool[idx]
                pop_idx = np.setdiff1d(np.arange(x_pool.shape[0]), idx)
                x_pool = x_pool[pop_idx, ...]
                y_train = y_train[pop_idx, ...]

        super().__init__(x_pool, x_test, y_test, classes, mask, label_function, y_train)
        self.exemplar_examples = exemplar_examples
        self.distances = {}
        self.eps = epsilon
        self.rv_bounds = rv_bounds if rv_bounds is not None else np.array([20.]*classes.shape[0])
        self.sigma = {}
        self.tolerance = 0.1
        self.opt_sig_n_draws = 50
        self.missing_classes = self.classes
        self.counts = []
        self.total_steps = 0
        self.min_prob = min_prob
        self.thompson = thompson
        self.compute_distances()
        self.optimize_sigma()
        self.weighted_samples = np.zeros((0, classes.shape[0]))
        self.samples = np.zeros((0, classes.shape[0]))
        tqdm.write(f"Classes: {classes}")

    def _compute_ess(self, weights):
        '''
        Compute effective sample size.

        See e.g. http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
        '''
        w = weights / weights.sum()
        return 1 / (w ** 2).sum()

    def _opt_sigma(self, sigma, dist, n_draws_star=25, test_draws=1000):
        if sigma < 1e-20:
            return np.inf
        ess = []
        n = self.x_pool.shape[0]
        p = np.exp(-(dist / (sigma))**2) + 1e-32
        p = p / p.sum()
        p = p.flatten()
        selection = np.random.choice(n, (test_draws, n_draws_star), replace=True, p=p)
        weights = (1/n * p**(-1) )[selection]
        w = weights / weights.sum(axis=1)[:,None]
        ess = (1/(w**2).sum(axis=1)).mean()
        return ess
    
    def compute_distances(self):
        for class_i in self.classes:
            feat = self.exemplar_examples[class_i]
            self.distances[class_i] = euclidean_distances(self.x_pool, feat[None,:])
    
    def optimize_sigma(self):
        for class_i in self.classes:
            f = lambda x: self._opt_sigma(x, 
                                        dist=self.distances[class_i],  
                                        n_draws_star=self.opt_sig_n_draws)
            sigma = minimize_scalar(f, tol=self.tolerance).x
            self.sigma[class_i] = sigma
            tqdm.write(f"Class {class_i}, sigma: {sigma}")

    def sample_with_kernel(self, p, n_draws):
        weights = []
        labels = []
        selected = []
        n = p.shape[0]
        for step in range(n_draws):
            done = False
            # while not done:
            selection = np.random.choice(p.shape[0], 100_000, replace=True, p=p)
            for j, s in enumerate(selection):
                if (j + 1) % 10_000 == 0:
                    print(f"{self.__class__.__name__}. Step {j+1}.")
                if s in self.x_pool_unlabeled:
                    self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, np.array([s]))
                    selected.append(s)
                    labels.append(self.label_function(s))
                    weights.append([(labels[-1] == t) * (1/n * (1/p[s])) for t in self.classes])
                    done = True
                    self.counts.append(self.total_steps+step)
                    break
                else:
                    # call the label function for free because we already have the label
                    # TODO: don't make an explicit call to the label function - probably the
                    # easiest way to handle this is internally in the label function (i.e. it 
                    # returns what it has returned before if the example has been queried before)
                    weights.append([(self.label_function(s) == t) * (1/n * (1/p[s])) for t in self.classes])
                    self.counts.append(self.total_steps+step)
            if not done:
                # select uniform at random if you can't find points
                print(f"{self.__class__.__name__}. Selecting uniform at random")
                selection = np.random.choice(p.shape[0], 1000, replace=True, p=1/p.shape[0] * np.ones_like(p))
                for j, s in enumerate(selection):
                    if s in self.x_pool_unlabeled:
                        self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, np.array([s]))
                        selected.append(s)
                        labels.append(self.label_function(s))
                        weights.append([(labels[-1] == t) * (1.) for t in self.classes])
                        done = True
                        self.counts.append(self.total_steps+step)
                        break
                    else:
                        # call the label function for free because we already have the label
                        # TODO: don't make an explicit call to the label function - probably the
                        # easiest way to handle this is internally in the label function (i.e. it 
                        # returns what it has returned before if the example has been queried before)
                        weights.append([(self.label_function(s) == t) * (1.) for t in self.classes])
                        self.counts.append(self.total_steps+step)
                
        self.weighted_samples = np.concatenate([self.weighted_samples, np.array(weights)], axis=0)
        x_samples = self.x_pool[np.array(selected), ...]
        y_samples = np.array(labels)
        self.total_steps += n_draws
        return x_samples, y_samples
    
    def update_score(self):
        try:
            p = self.model.predict_proba(self.x_pool[self.x_pool_unlabeled, ...])
            self.scores[self.x_pool_unlabeled] = -(1 - p.max(axis=1))
        except sklearn.exceptions.NotFittedError:
            self.scores[self.x_pool_unlabeled] = np.random.rand(self.x_pool_unlabeled.shape[0]) * 10 - 1000 # random scores
    
    def _prob(self, dist):
        return np.exp(dist - logsumexp(dist))

    def update_missing_classes(self):
        if self.thompson:
            estimated_p_y = self.weighted_samples.mean(axis=0)
            m = self.rv_bounds # TODO: automate the setting of m
            bounds = []
            n = np.maximum(self.weighted_samples.shape[0], 1)
            logs = []
            for j in self.missing_classes:
                j = int(j)
                z = self.weighted_samples[:, j] / m[j]
                bounds = np.array(emp_bernstein(z, delta=1 / n, n=n)) * m[j]
                logs.append(",".join([str(k) for k in [j, z.mean()*m[j], bounds[0], bounds[1]]]))
                # lower bound exceeds the threshold or upper bound is below the threshold
                if (bounds[0] > self.tolerance) or (bounds[1] < self.tolerance):
                    self.missing_classes = np.setdiff1d(self.missing_classes, np.array([j]))
                    tqdm.write(f"Removing class {j}")
        else:
            estimated_p_y = self.samples.mean(axis=0)
            n = np.maximum(self.samples.shape[0], 1)
            for j in self.missing_classes:
                j = int(j)
                bounds = chernoff(self.samples[:,j], 1./(1+self.samples.shape[0]))
                # lower bound exceeds the threshold or upper bound is below the threshold
                if (bounds[0] > self.tolerance) or (bounds[1] < self.tolerance):
                    self.missing_classes = np.setdiff1d(self.missing_classes, np.array([j]))
                    tqdm.write(f"Removing class {j}")

    def _random_select(self, n):
        full_idx = np.random.permutation(self.x_pool_unlabeled)
        i = full_idx[0:n]
        self.x_pool_unlabeled = full_idx[n:]
        x = self.x_pool[i, ...]
        y = np.array(self.label_function(i))
        samples = np.zeros((y.shape[0], len(self.classes)))
        for i, t in enumerate(self.classes):
            samples[:, i] = 1. * (y == t)
        self.samples = np.concatenate([self.samples, samples], axis=0)
        return x, y

    def _greedy_select(self, n, scores):
        idx = np.argsort(scores.flatten())[0:n]
        select = self.x_pool_unlabeled[idx]
        self.x_pool_unlabeled = np.setdiff1d(self.x_pool_unlabeled, select)
        labels = self.label_function(select)
        self.missing_classes = np.setdiff1d(self.missing_classes, np.unique(labels))
        return self.x_pool[select, ...], labels

    def _append_and_update(self, x, y, old_x, old_y):
        old_x = np.concatenate([old_x, x], axis=0)
        old_y = np.concatenate([old_y, y], axis=0)
        self.missing_classes = np.setdiff1d(self.missing_classes, np.unique(y))
        self.update_missing_classes()
        return old_x, old_y

    def _select_points(self, n=1):
        x_samples = np.zeros((0, self.x_pool.shape[1]))
        y_samples = np.zeros((0))
        if len(self.missing_classes) > 0:
            tqdm.write(f"Missing classes: {self.missing_classes}")
        step_count = 0
        if len(self.missing_classes) > 1:
            missing_classes = np.random.permutation(self.missing_classes.copy())
            while len(missing_classes) > 0:
                class_i = missing_classes[0]
                missing_classes = missing_classes[1:]
                dist = self.distances[class_i]
                sigma = self.sigma[class_i]
                n_samples = n // len(self.classes)
                if n_samples == 0:
                    warnings.warn(f"Skipping class {class_i} - only {n_samples}/{n} allocated")
                    continue
                if self.thompson:
                    p = self._prob(-self.distances[class_i] / self.sigma[class_i]).flatten()
                    p += self.min_prob # ensure weights don't blow up
                    p /= p.sum()
                    x, y = self.sample_with_kernel(p, n_samples)
                else:
                    n_random = (np.random.rand(n_samples) < self.eps).sum()
                    if n_random > 0:
                        x, y = self._random_select(n_random)
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
                missing_classes = np.setdiff1d(missing_classes, np.unique(y))
            step_count += x_samples.shape[0]
        if step_count < n:
            self.update_score()
            if self.thompson:
                # thompson sampling
                x, y = self._thompson_select(n - step_count, 10.)
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
            else:
                # epsilon greedy
                done = len(self.missing_classes) == 0 # TODO: replace with stopping rule
                n_random = 0 if done else (np.random.rand(n - step_count) < self.eps).sum()
                if n_random > 0:
                    x, y = self._random_select(n_random)
                    x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)

                x, y = self._greedy_select(scores=self.scores[self.x_pool_unlabeled], 
                                           n=n-x_samples.shape[0])
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
        return x_samples, y_samples

class ThompsonLearner(ImportanceLearner):
    def __init__(self, x_pool, x_test, y_test, exemplar_examples=None, classes=None, mask=None, label_function=None, y_train=None, 
                 random_example=True, min_prob=1e-6, rv_bounds=None):
        classes = classes if classes is not None else np.unique(y_train)
        if exemplar_examples is None:
            exemplar_examples = {}
            for i in classes:
                if random_example:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    idx = indices[np.random.randint(indices.shape[0])]
                else:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    x_ave = np.mean(x_pool[indices, ...], axis=0)
                    dist = euclidean_distances(x_pool[indices, ...], x_ave[None, :])
                    sorted_idx = np.argsort(dist)
                    idx = indices[sorted_idx[0]][0]
                exemplar_examples[i] = x_pool[idx]
                pop_idx = np.setdiff1d(np.arange(x_pool.shape[0]), idx)
                x_pool = x_pool[pop_idx, ...]
                y_train = y_train[pop_idx, ...]

        ActiveLearner.__init__(self, x_pool, x_test, y_test, classes, mask, label_function, y_train)
        self.exemplar_examples = exemplar_examples
        self.distances = {}
        self.rv_bounds = rv_bounds if rv_bounds is not None else np.array([20.]*classes.shape[0])
        self.sigma = {}
        self.tolerance = 0.1
        self.opt_sig_n_draws = 50
        self.missing_classes = self.classes
        self.counts = []
        self.total_steps = 0
        self.min_prob = min_prob
        self.compute_distances()
        self.optimize_sigma()
        self.weighted_samples = np.zeros((0, classes.shape[0]))
        tqdm.write(f"Classes: {classes}")

    def update_missing_classes(self):
        estimated_p_y = self.weighted_samples.mean(axis=0)
        m = self.rv_bounds # TODO: automate the setting of m
        bounds = []
        n = np.maximum(self.weighted_samples.shape[0], 1)
        for j in self.missing_classes:
            j = int(j)
            z = self.weighted_samples[:, j] / m[j]
            bounds = np.array(emp_bernstein(z, delta=1 / n, n=n)) * m[j]
            # lower bound exceeds the threshold or upper bound is below the threshold
            if (bounds[0] > self.tolerance) or (bounds[1] < self.tolerance):
                self.missing_classes = np.setdiff1d(self.missing_classes, np.array([j]))
                tqdm.write(f"Removing class {j}")

    def _thompson_select(self, n, scale=10.):
        x_samples = y_samples = None
        for step in range(n):
            p = self._prob(-self.scores[self.x_pool_unlabeled] * scale).flatten()
            x, y = self.sample_with_kernel(p, 1)
            x_samples = np.concatenate([x_samples, x], axis=0) if x_samples is not None else x
            y_samples = np.concatenate([y_samples, y], axis=0) if y_samples is not None else y
        return x_samples, y_samples
    
    def _select_points(self, n=1):
        x_samples = np.zeros((0, self.x_pool.shape[1]))
        y_samples = np.zeros((0))
        if len(self.missing_classes) > 0:
            tqdm.write(f"Missing classes: {self.missing_classes}")
        step_count = 0
        if len(self.missing_classes) > 1:
            missing_classes = np.random.permutation(self.missing_classes.copy())
            while len(missing_classes) > 0:
                class_i = missing_classes[0]
                missing_classes = missing_classes[1:]
                dist = self.distances[class_i]
                sigma = self.sigma[class_i]
                n_samples = n // len(self.classes)
                if n_samples == 0:
                    warnings.warn(f"Skipping class {class_i} - only {n_samples}/{n} allocated")
                    continue

                p = self._prob(-(self.distances[class_i] / self.sigma[class_i])**2).flatten()
                p += self.min_prob # ensure weights don't blow up
                p /= p.sum()
                x, y = self.sample_with_kernel(p, n_samples)

                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
                missing_classes = np.setdiff1d(missing_classes, np.unique(y))
            step_count += x_samples.shape[0]
        if step_count < n:
            self.update_score()
            x, y = self._thompson_select(n - step_count, scale=10.)
            x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)

        return x_samples, y_samples

class EPSGreedyLearner(ImportanceLearner):
    def __init__(self, x_pool, x_test, y_test, exemplar_examples=None, classes=None, mask=None, label_function=None, y_train=None, 
                 random_example=True, min_prob=1e-6, epsilon=0.1, **kwargs):
        classes = classes if classes is not None else np.unique(y_train)
        if exemplar_examples is None:
            exemplar_examples = {}
            for i in classes:
                if random_example:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    idx = indices[np.random.randint(indices.shape[0])]
                else:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    x_ave = np.mean(x_pool[indices, ...], axis=0)
                    dist = euclidean_distances(x_pool[indices, ...], x_ave[None, :])
                    sorted_idx = np.argsort(dist)
                    idx = indices[sorted_idx[0]][0]
                exemplar_examples[i] = x_pool[idx]
                pop_idx = np.setdiff1d(np.arange(x_pool.shape[0]), idx)
                x_pool = x_pool[pop_idx, ...]
                y_train = y_train[pop_idx, ...]
        ActiveLearner.__init__(self, x_pool, x_test, y_test, classes, mask, label_function, y_train)
        self.exemplar_examples = exemplar_examples
        self.eps = epsilon
        self.distances = {}
        self.tolerance = 0.1
        self.missing_classes = self.classes
        self.counts = []
        self.total_steps = 0
        self.min_prob = min_prob
        self.compute_distances()
        self.samples = np.zeros((0, classes.shape[0]))
        tqdm.write(f"Classes: {classes}")

    def update_missing_classes(self):
        estimated_p_y = self.samples.mean(axis=0)
        n = np.maximum(self.samples.shape[0], 1)
        for j in self.missing_classes:
            bounds = chernoff(self.samples[:,int(j)], 1./(1+self.samples.shape[0]))
            # lower bound exceeds the threshold or upper bound is below the threshold
            if (bounds[0] > self.tolerance) or (bounds[1] < self.tolerance):
                self.missing_classes = np.setdiff1d(self.missing_classes, np.array([j]))
                tqdm.write(f"Removing class {j}")
    
    def _select_points(self, n=1):
        x_samples = np.zeros((0, self.x_pool.shape[1]))
        y_samples = np.zeros((0))
        if len(self.missing_classes) > 0:
            tqdm.write(f"Missing classes: {self.missing_classes}")
        step_count = 0
        if len(self.missing_classes) > 1:
            # guided learning
            missing_classes = np.random.permutation(self.missing_classes.copy())
            while len(missing_classes) > 0:
                class_i = missing_classes[0]
                missing_classes = missing_classes[1:]
                n_samples = n // len(self.classes)
                if n_samples == 0:
                    warnings.warn(f"Skipping class {class_i} - only {n_samples}/{n} allocated")
                    continue
                n_random = (np.random.rand(n_samples) < self.eps).sum()
                if n_random > 0:
                    # epsilon step
                    x, y = self._random_select(n_random)
                    x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
                
                if (n_samples - n_random) > 0:
                    # greedy step
                    x, y = self._greedy_select(scores=-self.distances[class_i][self.x_pool_unlabeled], 
                                               n=n_samples-n_random)
                    x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
                
                missing_classes = np.setdiff1d(missing_classes, np.unique(y))
            step_count += x_samples.shape[0]
        if step_count < n:
            # active learning
            self.update_score()
            # epsilon greedy
            done = len(self.missing_classes) == 0 # TODO: replace with stopping rule
            n_random = 0 if done else (np.random.rand(n - step_count) < self.eps).sum()
            if n_random > 0:
                # epsilon
                x, y = self._random_select(n_random)
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
            if (n-x_samples.shape[0]) > 0:
                # greedy
                x, y = self._greedy_select(scores=self.scores[self.x_pool_unlabeled], 
                                            n=n-x_samples.shape[0])
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
        return x_samples, y_samples

class HybridLearner(ImportanceLearner):    
    def _select_points(self, n=1):
        x_samples = np.zeros((0, self.x_pool.shape[1]))
        y_samples = np.zeros((0))
        if len(self.missing_classes) > 0:
            tqdm.write(f"Missing classes: {self.missing_classes}")
        step_count = 0
        if len(self.missing_classes) > 1:
            # guided learning
            missing_classes = np.random.permutation(self.missing_classes.copy())
            while len(missing_classes) > 0:
                class_i = missing_classes[0]
                missing_classes = missing_classes[1:]
                dist = self.distances[class_i]
                sigma = self.sigma[class_i]
                n_samples = n // len(self.classes)
                if n_samples == 0:
                    warnings.warn(f"Skipping class {class_i} - only {n_samples}/{n} allocated")
                    continue

                p = self._prob(-(self.distances[class_i] / self.sigma[class_i])**2).flatten()
                p += self.min_prob # ensure weights don't blow up
                p /= p.sum()
                x, y = self.sample_with_kernel(p, n_samples)

                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
                missing_classes = np.setdiff1d(missing_classes, np.unique(y))
            step_count += x_samples.shape[0]
        if step_count < n:
            # active learning
            self.update_score()
            # epsilon greedy
            done = len(self.missing_classes) == 0 # TODO: replace with stopping rule
            n_random = 0 if done else (np.random.rand(n - step_count) < self.eps).sum()
            if n_random > 0:
                # epsilon
                x, y = self._random_select(n_random)
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
            if (n-x_samples.shape[0]) > 0:
                # greedy
                x, y = self._greedy_select(scores=self.scores[self.x_pool_unlabeled], 
                                            n=n-x_samples.shape[0])
                x_samples, y_samples = self._append_and_update(x,y,x_samples,y_samples)
        return x_samples, y_samples

def plot(experiments, models, names, exp_name="test"):
    '''
    Plotting code to produce training figures
    '''
    fig, ax = plt.subplots(figsize=(16,16))
    for j, metric in enumerate(["imbalance", "accuracy", "class_coverage"]):
        for i, (m, name) in enumerate(zip(models, names)):
            x = np.array([e[i]["queries"] for e in experiments]).mean(axis=0)
            mean =np.array([e[i][metric] for e in experiments]).mean(axis=0)
            sd =np.array([e[i][metric] for e in experiments]).std(axis=0) * 1.96 / np.sqrt(len(experiments))
            ax.plot(x, mean, color=f"C{i}", label=name)
            ax.fill_between(x, mean + sd, mean - sd, color=f"C{i}", alpha=0.1)
            ax.set_title(metric.replace("_"," ").title())
            ax.legend()
        fig.savefig(f"results/plots/{metric}-{exp_name}.png")
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_rare', type=int, default=50)
    parser.add_argument('--prop_rare', type=float, default=None)
    parser.add_argument('--dataset', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    n_seeds = 100 # number of times the experiment is run
    n_samples = args.n_samples # total number of active learning samples labeled
    batch_size = args.batch_size # number of samples labeled before updating the active learning model
    dataset = args.dataset
    n_rare = args.n_rare
    exp_name = f"supp_{dataset}_rare-{n_rare}_rare-{args.prop_rare}_bs-{batch_size}_samples-{n_samples}"

    # set of models to experiment with
    experiment_methods = {"EGAL hybrid": HybridLearner,
                          "EGAL importance": ThompsonLearner,
                          "EGAL eps-greedy": EPSGreedyLearner,
                          "Least Confidence":LeastConfidence,
                          "Guided Learner":GuidedLearner,
                          "Random Search":RandomSearch,
                          "Entropy Search": EntropySearch
                        }
    experiments = []
    # run the experiment n_seeds times to test variance across experiments
    for exp_num in tqdm(range(n_seeds)):
        X_train, y_train, X_test, y_test, exemplar_examples, classes = get_data(dataset, n_rare=n_rare, prop_rare=args.prop_rare)
        exemplar_examples = None

        if exemplar_examples is None:
            #classes = np.unique(y_train)
            exemplar_examples = {}
            for i in classes:
                if (y_train == i).sum() != 0:
                    indices = np.arange(y_train.shape[0])[y_train == i]
                    idx = indices[np.random.randint(indices.shape[0])]
                    exemplar_examples[i] = X_train[idx]
                    pop_idx = np.setdiff1d(np.arange(X_train.shape[0]), idx)
                    X_train = X_train[pop_idx, ...]
                    y_train = y_train[pop_idx, ...]
                else:
                    exemplar_examples[i] = X_train.mean(axis=0) + np.random.randn(X_train.shape[1]) * X_train.std(axis=0)

            print(exemplar_examples)

        models = {k: c(X_train, X_test, y_test, y_train=y_train, 
                       exemplar_examples=exemplar_examples, classes=classes, rv_bounds=[30.]*10) 
                  for k, c in experiment_methods.items()}
        
        # main training loop
        for _ in tqdm(range(n_samples//batch_size)):
            # for each active learning algorithm, select batch_size examples according to the
            # aquisition proceedure and update the active learning model
            [m.select_and_train(batch_size) for k, m in models.items()]
        
        # save performance and print / plot partial results
        experiments.append([m.perf for k, m in models.items()])
        plotperf(experiments, models, experiment_methods.keys(), exp_name=exp_name, pdf=(exp_num % 10)==0)

    # save performance and plot final results
    names = experiment_methods.keys()
    assert len(names) == len(models)
    fig, ax = plt.subplots(2,2, figsize=(16,16))
    for j, metric in enumerate(["imbalance", "accuracy", "class_coverage", "normalized-accuracy"]):
        if metric == "normalized-accuracy":
            normalized = True
            metric = "accuracy"
        else:
            normalized = False
        for i, (m, name) in enumerate(zip(models, names)):
            a = j // 2
            b = j % 2
            x = np.array([e[i]["queries"] for e in experiments]).mean(axis=0)
            mean =np.array([e[i][metric] for e in experiments]).mean(axis=0)
            sd =np.array([e[i][metric] for e in experiments]).std(axis=0) * 1.96 / np.sqrt(len(experiments))
            ax[a,b].plot(x, mean, color=f"C{i}", label=name)
            ax[a,b].fill_between(x, mean + sd, mean - sd, color=f"C{i}", alpha=0.1)
            ax[a,b].set_title(metric.replace("_"," ").title())
            ax[a,b].legend()
    fig.savefig(f"results/plots/{exp_name}.png")
    fig.savefig(f"results/plots/{exp_name}.pdf")
    pickle.dump(experiments, open(f"results/dumps/{exp_name}.pkl", "wb"))

if __name__ == "__main__":
    main()
