import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, HeNormal
import tensorflow as tf
tf.config.run_functions_eagerly(True)

def get_available_devices():
    """Returns a list of available GPU devices, falling back to CPU if none available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    return gpus

class NN_base:
    def __init__(
            self,
            learning_rate,
            output_activation,
            loss_function,
            weight_init,
            num_epochs,
            class_weights,
            minibatch_size,
            L2_lambda,
            flag_batchnorm,
            seed,
            device=None
    ):
        # Set random seed
        tf.random.set_seed(seed)
        self.seed = seed
        
        # Device management
        self.device = device
        if self.device is None:
            gpus = get_available_devices()
            self.device = '/GPU:0' if gpus else '/CPU:0'
        
        # NN parameters
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.num_epochs = num_epochs
        self.class_weights = class_weights
        self.minibatch_size = minibatch_size
        self.L2_lambda = L2_lambda
        self.flag_batchnorm = flag_batchnorm
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    

        # Weight initialization
        if weight_init == "glorot":
            self.weight_init = GlorotUniform(seed=self.seed)
        elif weight_init == "he":
            self.weight_init = HeNormal(seed=self.seed)

        # Loss function
        self.loss_function = loss_function

        # Model to be defined in sub-classes
        self.model = None

    def cast_classes(self, y_datasets):
        return [tf.cast(y_data, tf.int32) for y_data in y_datasets]

    @tf.function(jit_compile=True)
    def predict(self, x):
        with tf.device(self.device):
            y_hat = self.model(x, training=False)
            y_hat_class = tf.cast(y_hat > 0.5, tf.int32)
            return y_hat.numpy(), y_hat_class.numpy()

    @tf.function(jit_compile=True)
    def training(self, x, y):
        with tf.device(self.device):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = self.loss_function(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def change_class_weights(self, weights):
        if self.class_weights != weights:
            self.class_weights = weights

    def get_num_epochs(self):
        return self.num_epochs

    def change_num_epochs(self, n_epochs):
        if self.num_epochs != n_epochs:
            self.num_epochs = n_epochs

    def change_minibatch_size(self, batch_size):
        if self.minibatch_size != batch_size:
            self.minibatch_size = batch_size