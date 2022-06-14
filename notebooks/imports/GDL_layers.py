import tensorflow as tf
import tensorflow.experimental.numpy as tfnp


# Layer that implements 4 discrete rotational equivariance
# Note that this uses group convolution on Z^2 x C_4, where rotation is about the center of the domain.
class RotEquivConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            out_features,
            filt_shape, 
            rot_axis=True,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform,
            bias_initializer=tf.keras.initializers.zeros,
            **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.filt_shape = filt_shape
        self.rot_axis = rot_axis
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer()
        if use_bias:
            self.bias_initializer = bias_initializer()

    def build(self, input_shape):  # Create the layer when it is first called
        self.in_features = input_shape[-1]
        self.filt_shape = tf.concat([
            self.filt_shape,  # Spatial dimensions
            [self.in_features, self.out_features]
        ], axis=0)
        self.filt_base = tf.Variable(
            self.kernel_initializer(self.filt_shape),  # Random initialization of filters
            name='kernel'
        )
        if self.use_bias:
            self.bias = tf.Variable(self.bias_initializer((self.out_features,)), name='bias')

    def call(self, inputs):  # Does the actual computation for each rotation
        if self.rot_axis:  # If we're already in Z^2 x C_4, convolve along each rotational layer
            outputs = self.activation(tf.stack([
                    tf.nn.convolution(
                        tfnp.take(inputs, i, axis=-2),
                        self.filt_base)
                    for i in range(inputs.shape[-2])],
                axis=-2
            ))
        else:  # If we're not yet in the group domain, move to it.
            outputs = self.activation(tf.stack([
                tf.nn.convolution(inputs, self.filt_base),
                tfnp.rot90(tf.nn.convolution(inputs, tfnp.rot90(self.filt_base, k=1)), k=4-1, axes=(1, 2)),
                tfnp.rot90(tf.nn.convolution(inputs, tfnp.rot90(self.filt_base, k=2)), k=4-2, axes=(1, 2)),
                tfnp.rot90(tf.nn.convolution(inputs, tfnp.rot90(self.filt_base, k=3)), k=4-3, axes=(1, 2)),
            ], axis=-2))
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


# 2D pooling layer that pools within each rotational dimension
class RotEquivPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size, pool_method=tf.keras.layers.MaxPool2D, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_method = pool_method
        self.pool = self.pool_method(pool_size=self.pool_size)

    def call(self, inputs):
        return tf.stack(
            [self.pool(tfnp.take(inputs, k, axis=-2)) for k in range(inputs.shape[-2])],
            axis=-2
        )


# Rotational invariant pooling that pools across the rotational dimensions
class RotInvPool(tf.keras.layers.Layer):
    def __init__(self, pool_method='max', **kwargs):
        valid_methods = {'max', 'mean'}
        if pool_method not in valid_methods:
            raise ValueError(f'pool_method must be one of {valid_methods}')

        super().__init__(**kwargs)
        if pool_method == "max":
            self.pool_method = tf.math.reduce_max
        else:
            self.pool_method = tf.math.reduce_mean

    def call(self, inputs):
        return self.pool_method(inputs, axis=-2)