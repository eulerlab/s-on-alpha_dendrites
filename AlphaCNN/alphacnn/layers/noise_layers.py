import tensorflow as tf
from keras import layers
from keras.backend import floatx
from keras.src.backend import RandomGenerator
from keras.src.utils import tf_utils
from keras.src.utils.version_utils import base_layer
# isort: off
from tensorflow.python.util.tf_export import keras_export


class NeuroRandomGenerator(RandomGenerator):
    def random_poisson(
            self, shape, lam, dtype=None, nonce=None
    ):
        """Produce random number based on the normal distribution.
        Args:
          shape: The shape of the random values to generate.
          lam: Floats. Lambda of the random values to generate.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.poisson(
                shape=shape, lam=lam, dtype=dtype
            )
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)

            return tf.random.stateless_poisson(
                shape=shape, lam=lam, dtype=dtype, seed=seed
            )

        outputs = tf.random.poisson(
            shape=(1,),
            lam=lam,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )

        return outputs


@keras_export("keras.__internal__.layers.NeuroBaseRandomLayer")
class NeuroBaseRandomLayer(base_layer.BaseRandomLayer):
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(
            self, seed=None, force_generator=False, rng_type=None, **kwargs
    ):
        """Overwrite engine.BaseRandomLayer to use custom noise"""
        super().__init__(**kwargs)
        self._random_generator = NeuroRandomGenerator(
            seed, force_generator=force_generator, rng_type=rng_type
        )


@keras_export("keras.layers.ActiveGaussianNoise")
class GaussianNoiseWithSameShape(layers.GaussianNoise):
    def call(self, inputs, training=None):
        def noised():
            return self._random_generator.random_normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=inputs.dtype,
            )

        return noised()


@keras_export("keras.layers.ActiveAddGaussianNoise")
class ActiveGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        def noised():
            return inputs + self._random_generator.random_normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=inputs.dtype,
            )

        return noised()


@keras_export("keras.layers.PoissonNoise")
class ActivePoissonNoise(NeuroBaseRandomLayer):
    """Apply Poisson noise.
    Args:
      stddev: Float, standard deviation of the noise distribution.
      seed: Integer, optional random seed to enable deterministic behavior.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a response.
    Output shape:
      Same shape as input.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.seed = seed

    def call(self, inputs, training=None):
        def noised():
            return self._random_generator.random_poisson(
                shape=tf.shape(inputs),
                lam=inputs,
                dtype=inputs.dtype,
            )

        return noised()

    def get_config(self):
        config = {"seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
