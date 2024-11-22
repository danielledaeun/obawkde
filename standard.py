import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from base import NN_base

class NN_standard(NN_base):
    def __init__(
            self,
            layer_dims,
            learning_rate,
            output_activation,
            loss_function,
            weight_init,
            num_epochs,
            class_weights,
            minibatch_size,
            L2_lambda=0.0,
            flag_batchnorm=False,
            seed=0,
            device=None
    ):
        super().__init__(
            learning_rate=learning_rate,
            output_activation=output_activation,
            loss_function=loss_function,
            weight_init=weight_init,
            num_epochs=num_epochs,
            class_weights=class_weights,
            minibatch_size=minibatch_size,
            L2_lambda=L2_lambda,
            flag_batchnorm=flag_batchnorm,
            seed=seed,
            device=device
        )

        self.layer_dims = layer_dims
        self.model = self.create_standard_model()
        
        # Configure model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.loss_function,
            metrics=['accuracy']
        )

    def create_standard_model(self):
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]

        # Input layer
        X_input = Input(shape=(n_x,), name='input')
        X = X_input

        # Hidden layers
        for i, units in enumerate(self.layer_dims[1:-1], 1):
            X = Dense(
                units=units,
                use_bias=True,
                kernel_initializer=self.weight_init,
                bias_initializer='zeros',
                kernel_regularizer=l2(self.L2_lambda)
            )(X)
            
            if self.flag_batchnorm:
                X = BatchNormalization()(X)
            X = LeakyReLU(negative_slope=0.01)(X)

        # Output layer
        y_out = Dense(
            units=n_y,
            activation=self.output_activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            name='output'
        )(X)

        return Model(inputs=X_input, outputs=y_out)