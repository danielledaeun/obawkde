import numpy as np
from collections import deque
from collections import Counter
import tensorflow as tf

class Baseline:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, y):
        self.model.training(x, y)

class Sliding(Baseline):
    def __init__(self, model, sliding_window_size):
        super().__init__(model)
        self.sliding_window_size = sliding_window_size
        self.xs = deque(maxlen=self.sliding_window_size)
        self.ys = deque(maxlen=self.sliding_window_size)

    def append_to_win(self, x, y, n_features):
        # Append to queues
        self.xs.append(x)
        self.ys.append(y)

        # Update batch size
        size = len(self.ys)
        self.model.change_minibatch_size(size)

        # Convert to tensors
        x = tf.convert_to_tensor(
            np.array(self.xs).reshape(size, n_features), 
            dtype=tf.float32
        )
        y = tf.convert_to_tensor(
            np.array(self.ys).reshape(size, 1), 
            dtype=tf.float32
        )

        return x.numpy(), y.numpy()

class OOB:
    def __init__(self, models):
        self.models = models
        self.n_models = len(models)
        self.epoch_counts = np.ones(self.n_models, dtype=int)  # Initialize with 1 epoch each

    def predict(self, x):
        # Get predictions from all models
        preds = [m.predict(x)[0] for m in self.models]
        y_hats = [a.flatten()[0] for a in preds]

        # Average vote
        y_hats_avg = np.mean(y_hats).reshape(1, 1)

        # Majority vote (useful for multi-model ensemble)
        rounded = [round(float(x)) for x in y_hats]
        y_hats_avg_class = max(rounded, key=Counter(rounded).get)
        y_hats_avg_class = np.reshape(np.array(y_hats_avg_class), (1, 1))

        return y_hats_avg, y_hats_avg_class

    def train(self, x, y):
        for i, (model, n_epochs) in enumerate(zip(self.models, self.epoch_counts)):
            # Train in one go instead of loop
            for _ in range(n_epochs):
                model.training(x, y)
                
    def oob_oversample(self, random_state, imbalance_rate):
        # Ensure imbalance_rate is a scalar
        imbalance_rate = float(imbalance_rate)
        # Generate poisson samples with proper shape
        poisson_samples = random_state.poisson(lam=imbalance_rate, size=(self.n_models,))
        self.epoch_counts = np.maximum(1, poisson_samples).astype(np.int32)

class AREBA(Baseline):
    def __init__(self, model, queue_size_budget):
        super().__init__(model)
        
        # budget
        self.budget = queue_size_budget

        # init queues
        self.xs_neg = deque(maxlen=1)
        self.ys_neg = deque(maxlen=1)
        self.xs_pos = deque(maxlen=1)
        self.ys_pos = deque(maxlen=1)

    def adapt_queue(self, q, q_cap):
        if q == 'neg':
            self.xs_neg = deque(self.xs_neg, q_cap)
            self.ys_neg = deque(self.ys_neg, q_cap)
        elif q == 'pos':
            self.xs_pos = deque(self.xs_pos, q_cap)
            self.ys_pos = deque(self.ys_pos, q_cap)

    def get_training_set(self, n_features):
        # Merge queues
        xs = list(self.xs_neg) + list(self.xs_pos)
        ys = list(self.ys_neg) + list(self.ys_pos)

        # Convert to tensors
        size = len(ys)
        x = tf.convert_to_tensor(
            np.array(xs).reshape(size, n_features), 
            dtype=tf.float32
        )
        y = tf.convert_to_tensor(
            np.array(ys).reshape(size, 1), 
            dtype=tf.float32
        )

        # Update batch size
        self.model.change_minibatch_size(size)

        return x.numpy(), y.numpy()

    def append_to_queues(self, x, y):
        if y == 0:
            self.xs_neg.append(x)
            self.ys_neg.append(y)
        else:
            self.xs_pos.append(x)
            self.ys_pos.append(y)

    def adapt_queues(self, delayed_size_neg, delayed_size_pos):
        length_q_pos = len(self.ys_pos)
        capacity_q_pos = self.ys_pos.maxlen
        length_q_neg = len(self.ys_neg)
        capacity_q_neg = self.ys_neg.maxlen

        if length_q_pos == 0 and capacity_q_neg < self.budget:
            self.adapt_queue('neg', capacity_q_neg + 1)
        elif length_q_neg == 0 and capacity_q_pos < self.budget:
            self.adapt_queue('pos', capacity_q_pos + 1)
        else:
            if delayed_size_neg > delayed_size_pos:
                if capacity_q_pos == length_q_pos:
                    if capacity_q_pos < self.budget / 2.0:
                        self.adapt_queue('pos', capacity_q_pos + 1)
                        self.adapt_queue('neg', capacity_q_pos)
                    elif capacity_q_pos == self.budget / 2.0 and capacity_q_neg != capacity_q_pos:
                        self.adapt_queue('neg', capacity_q_pos)

            if delayed_size_neg <= delayed_size_pos:
                if capacity_q_neg == length_q_neg:
                    if capacity_q_neg < self.budget / 2.0:
                        self.adapt_queue('neg', capacity_q_neg + 1)
                        self.adapt_queue('pos', capacity_q_neg)
                    elif capacity_q_neg == self.budget / 2.0 and capacity_q_pos != capacity_q_neg:
                        self.adapt_queue('pos', capacity_q_neg)