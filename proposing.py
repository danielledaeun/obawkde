import numpy as np
from collections import deque
import sys
from pathlib import Path
from typing import List, Tuple, Union

lib_dir = Path(__file__).parent / 'lib' 
sys.path.append(str(lib_dir))
from awkde import awkde

class OB_awKDE:
    """Online Biased Learning with Adaptive Weight Kernel Density Estimation.
    
    This class implements an online biased learning strategy using adaptive weight KDE
    for handling class imbalance in streaming data.
    """
    def __init__(self, models, queue_size):
        self.models = models
        self.queue_size = queue_size
        # init queues
        self.xs_neg = deque(maxlen=queue_size)
        self.ys_neg = deque(maxlen=queue_size)

        self.xs_pos = deque(maxlen=queue_size)
        self.ys_pos = deque(maxlen=queue_size)
        
        self.xs_kde = deque(maxlen=queue_size)
        self.ys_kde = deque(maxlen=queue_size)
        
    def append_to_queues(self, x, y):
        if y == 0:
            self.xs_neg.append(x)
            self.ys_neg.append(y)
        else:
            self.xs_pos.append(x)
            self.ys_pos.append(y)

    def change_kdesets(self, x, y):
        self.xs_kde = x       
        self.ys_kde = y            
    
    def kde_oversamping(self, target, n_features, k, seed):
        size_pos = len(list(self.ys_pos))  
        size_neg = len(list(self.ys_neg))  
        queue_size  = self.queue_size
        
        if (target==1) and (k < size_pos):
            x_res = np.array(list(self.xs_pos)).reshape(size_pos, n_features)
            y_kde = np.ones(queue_size).reshape(queue_size, 1)
            # kde
            kde_aw = awkde.GaussianKDE(glob_bw="silverman",alpha=0.5, diag_cov=True) #scott, silverman
            mean, cov = kde_aw.fit(x_res)
            x_kde = kde_aw.sample(n_samples=queue_size, random_state=seed).reshape(queue_size, n_features)
            self.change_kdesets(x_kde, y_kde)
        elif (target==0) and (k < size_neg) :
            x_res = np.array(list(self.xs_neg)).reshape(size_neg, n_features)
            y_kde = np.zeros(queue_size).reshape(queue_size, 1)
            # kde
            kde_aw = awkde.GaussianKDE(glob_bw="silverman",alpha=0.5, diag_cov=True) #scott, silverman
            mean, cov = kde_aw.fit(x_res)
            x_kde = kde_aw.sample(n_samples=queue_size, random_state=seed).reshape(queue_size, n_features)
            self.change_kdesets(x_kde, y_kde)
        else: 
            pass

    def get_training_set(self, n_features, resampling, seed):
        # merge queues
        xs = list(self.xs_neg) + list(self.xs_pos)
        ys = list(self.ys_neg) + list(self.ys_pos)

        # convert merged queues to np arrays
        size = len(ys)  # current queue size
        x = np.array(xs).reshape(size, n_features)
        y = np.array(ys).reshape(size, 1)

        ## add KDE oversampling sets
        size_pos = len(list(self.ys_pos))  
        size_neg = len(list(self.ys_neg))
        size_kde = len(list(self.ys_kde))
        
        x_kde = np.array(list(self.xs_kde)).reshape(size_kde, n_features)
        y_kde = np.array(list(self.ys_kde)).reshape(size_kde, 1)
        
        if (size_pos != size_neg) and (size_kde > 0):
            ## sample ##
            np.random.seed(seed)
            idx = np.random.randint(size_kde, size=abs(size_neg - size_pos))
            x_add = x_kde[idx,:]
            y_add = y_kde[idx,:]
            # merge
            x = np.r_[x, x_add]
            y = np.r_[y, y_add]
        
        # if equal and resampling
        elif (size_pos == size_neg) and (size_kde > 0) and resampling :
            np.random.seed(seed+1)
            idx = np.random.randint(size_kde, size=10)
            x_add = x_kde[idx,:]
            y_add = y_kde[idx,:]
            # merge
            x = np.r_[x, x_add]
            y = np.r_[y, y_add]
            
        return x, y
    
    # predict
    def predict(self, x):
        preds = [m.predict(x)[0] for m in self.models]
        y_hats = [a.flatten()[0] for a in preds]

        # average vote 
        y_hats_avg = np.mean(y_hats).reshape(1, 1)
        y_hats_avg_class = np.around(y_hats_avg)
        
        return y_hats_avg, y_hats_avg_class 

    # train
    def train(self, x, y, n_features,random_state):
        for i, m in enumerate(self.models):
            x, y = self.get_training_set(n_features, True, i)
            m.change_minibatch_size(len(y))
            m.training(x, y)  