from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import imports.GDL_layers as GDL_layers


class gdl_model(object):

    def __init__(
        self,
        filters=(32,),
        kernel_sizes=(3,),
        conv2d_activation='relu',
        conv2d_regularizer=None,
        pool_sizes=(4,),
        pool_dropout=0.0,
        latent_size=32,
        dense_activation=None,
        dense_regularizer=None,  
        dense_dropout=0.0,
        rot_inv=1,
        output_activation=None,
        output_regularizer=None,
        lr=0.001,
        optimizer="nadam",
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        sgd_momentum=0.9,
        decay=0,
        loss="mse",
        metrics=[],
        batch_size=32,
        epochs=10,
        verbose=0,
        **kwargs
    ):

        self.filters = filters
        self.kernel_sizes = [tuple((v, v)) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.conv2d_regularizer = conv2d_regularizer
        self.pool_sizes = [tuple((v, v)) for v in pool_sizes]
        self.pool_dropout = pool_dropout
        self.latent_size = latent_size
        self.dense_activation = dense_activation
        self.dense_regularizer = dense_regularizer
        self.dense_dropout = dense_dropout
        if not rot_inv:
            self.rot_inv = 9999
        else:
            self.rot_inv = rot_inv
        self.output_activation = output_activation
        self.output_regularizer = output_regularizer
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.sgd_momentum = sgd_momentum
        self.decay = decay
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def build_model(self, input_shape, output_shape):
        self.model = models.Sequential()

        self.model.add(
            GDL_layers.RotEquivConv2D(
                self.filters[0],
                self.kernel_sizes[0],
                rot_axis=False,
                input_shape=input_shape,
                activation=self.conv2d_activation,
                kernel_regularizer=self.conv2d_regularizer
            )
        )
        self.model.add(GDL_layers.RotEquivPool2D(self.pool_sizes[0]))
        if self.rot_inv == 1:
            self.model.add(
                GDL_layers.RotInvPool()
            )
        for h in range(1, len(self.filters)):
            if h < self.rot_inv:
                self.model.add(
                    GDL_layers.RotEquivConv2D(
                        self.filters[h],
                        self.kernel_sizes[h],
                        activation=self.conv2d_activation,
                        kernel_regularizer=self.conv2d_regularizer
                    )
                )
                self.model.add(
                    GDL_layers.RotEquivPool2D(self.pool_sizes[h])
                )
                if h+1 == self.rot_inv:
                    self.model.add(
                        GDL_layers.RotInvPool()
                    )
            else:
                self.model.add(
                    layers.Conv2D(
                        self.filters[h],
                        self.kernel_sizes[h],
                        activation=self.conv2d_activation,
                        kernel_regularizer=self.conv2d_regularizer
                    )
                )
                self.model.add(
                    layers.MaxPool2D(self.pool_sizes[h])
                )
        self.model.add(layers.Flatten())
        self.model.add(
            layers.Dense(
                self.latent_size,
                activation=self.dense_activation,
                kernel_regularizer=self.dense_regularizer
            )
        )
        self.model.add(
            layers.Dense(
                output_shape,
                activation=self.output_activation,
                kernel_regularizer=self.output_regularizer
            )
        )

        if self.optimizer == "nadam":
            self.optimizer_obj = tf.keras.optimizers.Nadam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
            )
        if self.optimizer == "adam":
            self.optimizer_obj = tf.keras.optimizers.Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
            )
        if self.optimizer == "SGD":
            self.optimizer_obj = tf.keras.optimizers.SGD(
                learning_rate=self.lr,
                momentum=self.sgd_momentum,
                decay=self.decay,
            )

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            metrics=self.metrics
        )

    def fit(self, x, y, xv=None, yv=None, callbacks=None):
        if len(x.shape[1:]) == 2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]

        input_shape = x.shape[1:]
        self.build_model(input_shape, output_shape)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(xv, yv), callbacks=callbacks)
        return self.model.history.history
    
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

class CNN_model(object):

    def __init__(
        self,
        filters=(32,),
        kernel_sizes=(3,),
        conv2d_activation='relu',
        conv2d_regularizer=None,
        pool_sizes=(4,),
        pool_dropout=0.0,
        latent_size=32,
        dense_activation=None,
        dense_regularizer=None,  
        dense_dropout=0.0,
        output_activation=None,
        output_regularizer=None,
        lr=0.001,
        optimizer="nadam",
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        sgd_momentum=0.9,
        decay=0,
        loss="mse",
        metrics=[],
        batch_size=32,
        epochs=10,
        verbose=0,
        **kwargs
    ):

        self.filters = filters
        self.kernel_sizes = [tuple((v, v)) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.conv2d_regularizer = conv2d_regularizer
        self.pool_sizes = [tuple((v, v)) for v in pool_sizes]
        self.pool_dropout = pool_dropout
        self.latent_size = latent_size
        self.dense_activation = dense_activation
        self.dense_regularizer = dense_regularizer
        self.dense_dropout = dense_dropout
        self.output_activation = output_activation
        self.output_regularizer = output_regularizer
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.sgd_momentum = sgd_momentum
        self.decay = decay
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def build_model(self, input_shape, output_shape):
        self.model = models.Sequential()
        
        self.model.add(
            layers.Conv2D(
                self.filters[0],
                self.kernel_sizes[0],
                activation=self.conv2d_activation,
                kernel_regularizer=self.conv2d_regularizer,
                input_shape=input_shape
            )
        )
        self.model.add(
            layers.MaxPooling2D(self.pool_sizes[0])
        )

        for h in range(1, len(self.filters)):
            self.model.add(
                layers.Conv2D(
                    self.filters[h],
                    self.kernel_sizes[h],
                    activation=self.conv2d_activation,
                    kernel_regularizer=self.conv2d_regularizer
                )
            )
            self.model.add(
                layers.MaxPooling2D(self.pool_sizes[h])
            )
        self.model.add(layers.Flatten())
        self.model.add(
            layers.Dense(
                self.latent_size,
                activation=self.dense_activation,
                kernel_regularizer=self.dense_regularizer
            )
        )
        self.model.add(
            layers.Dense(
                output_shape,
                activation=self.output_activation,
                kernel_regularizer=self.output_regularizer
            )
        )

        if self.optimizer == "nadam":
            self.optimizer_obj = tf.keras.optimizers.Nadam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
            )
        if self.optimizer == "adam":
            self.optimizer_obj = tf.keras.optimizers.Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
            )
        if self.optimizer == "SGD":
            self.optimizer_obj = tf.keras.optimizers.SGD(
                learning_rate=self.lr,
                momentum=self.sgd_momentum,
                decay=self.decay,
            )

        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss,
            metrics=self.metrics
        )

    def fit(self, x, y, xv=None, yv=None, callbacks=None):
        if len(x.shape[1:]) == 2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]

        input_shape = x.shape[1:]
        self.build_model(input_shape, output_shape)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(xv, yv), callbacks=callbacks)
        return self.model.history.history
    
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
