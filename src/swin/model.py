"""The Swin model definition module.

Attributes:
    DEFAULT_DROP_RATE: Default probability of dropping connections in 
        ``Dropout`` layers.
    DEFAULT_DROP_PATH_RATE: Default maximum probability of entirely skipping a 
        (Shifted) Windows Multi-head Attention computation (Stochastic Depth
        computation-skipping technique) during training.
        This maximum value is used in the last stage of the network, while
        previous stages use linearly spaced values in the 
        [0 ,``drop_path_rate``] interval.
"""

import collections.abc

import numpy as np
import tensorflow as tf

from swin.modules import SwinLinear, SwinPatchEmbeddings, SwinStage

DEFAULT_DROP_RATE: float = 0.0
DEFAULT_DROP_PATH_RATE: float = 0.1


@tf.keras.saving.register_keras_serializable(package=__package__)
class Swin(tf.keras.Model):
    """Swin Transformer Model.

    To stay consistent with the architecture described in the paper, this class
    assumes the following:

        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        patch_size: The size of patches in which images will be divided into.
            Expressed in pixels.
        window_size: The size of windows in (Shifted) Windows Multi-head
          Attention layers expressed in patches per axis.
        embed_dim: The length of embeddings built from patches.
        depths: The number of ``SwinTransformer`` layers in each stage of the
            network.
        num_heads: The number of (Shifted) Windows Multi-head Attention heads in
            each stage of the network.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
        drop_path_rate: The maximum probability of entirely skipping a (Shifted)
          Windows Multi-head Attention computation (Stochastic Depth
          computation-skipping technique) during training.
          This maximum value is used in the last stage of the network, while
          previous stages use linearly spaced values in the
          [0 ,``drop_path_rate``] interval.
    """

    def __init__(
        self,
        num_classes: int,
        patch_size: int = 4,
        window_size: int = 7,
        embed_dim: int = 96,
        depths: collections.abc.Collection[int] = (2, 2, 6, 2),
        num_heads: collections.abc.Collection[int] = (3, 6, 12, 24),
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(self.depths)
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.patch_embed = SwinPatchEmbeddings(
            self.embed_dim,
            self.patch_size,
            norm_layer=True,
            name="patches_linear_embedding",
        )

        self.pos_drop = tf.keras.layers.Dropout(rate=self.drop_rate)

        # We must use numpy to generate these values as using TensorFlow's
        # equivalent function would mean creating EagerTensor objects.
        # These tensor would then get returned as layer parameters through
        # calls to their get_config() methods, causing problems in the JSON
        # serialization as the built-in Python library cannot handle this
        # type of objects, thus preventing model saving.
        drop_depth_rate = [
            x
            for x in np.linspace(
                0.0,
                self.drop_path_rate,
                sum(depths),
            )
        ]

        self.blocks = tf.keras.Sequential(
            [
                SwinStage(
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=self.window_size,
                    mlp_ratio=4.0,
                    drop_p=drop_rate,
                    drop_path_p=drop_depth_rate[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=True if (i < self.num_layers - 1) else False,
                )
                for i in range(self.num_layers)
            ],
            name="swin_stages",
        )

        self.norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="layer_normalization"
        )
        self.head = SwinLinear(num_classes, name="classification_head")

    def build(self, input_shape: tf.TensorShape | list) -> None:
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2] and input_shape[3] == 3

        # Not mentioned anywhere in the docs, but the reference implementation
        # of build() for the Model class recursively builds all sub-layers and
        # sets the built attribute to True, so it must be called in order to
        # truly build the model. Failure to do so results in a broken model
        # when loading it back from disk, as layers will miss weights defined
        # in their build() methods.
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.patch_embed(inputs, **kwargs)
        x = self.pos_drop(x, **kwargs)
        x = self.blocks(x, **kwargs)

        x = self.norm(x, **kwargs)
        # We average along height and width, while simultaneously flattening
        # the result into (batch_size, embed_dim)
        x = tf.reduce_mean(x, [1, 2], name="average_pooling_and_flattening")
        x = self.head(x, **kwargs)
        x = tf.nn.softmax(x)

        return x

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "patch_size": self.patch_size,
                "window_size": self.window_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )

        return config

    def __repr__(self) -> str:
        return f"{Swin.__name__}(num_classes={self.num_classes}, patch_size={self.patch_size}, window_size={self.window_size}, embed_dim={self.embed_dim}, depths={self.depths}, num_heads={self.num_heads}, drop_rate={self.drop_rate}, drop_path_rate={self.drop_path_rate})"


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinT(Swin):
    """Swin-T Transformer Model.

    This version (tiny) uses the following options:

        - ``patch_size`` = 4
        - ``window_size`` = 7
        - ``embed_dim`` = 96
        - ``depths`` = (2, 2, 6, 2)
        - ``num_heads`` = (3, 6, 12, 24)

    To stay consistent with the architecture described in the paper, this class
    assumes the following:

        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
        drop_path_rate: The maximum probability of entirely skipping a (Shifted)
          Windows Multi-head Attention computation (Stochastic Depth
          computation-skipping technique) during training.
          This maximum value is used in the last stage of the network, while
          previous stages use linearly spaced values in the
          [0 ,``drop_path_rate``] interval.
    """

    def __init__(
        self,
        num_classes: int,
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )

        return config

    @classmethod
    def from_config(cls, config: dict) -> "SwinT":
        return Swin(**config)


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinS(Swin):
    """Swin-S Transformer Model.

    This version (small) uses the following options:

        - ``patch_size`` = 4
        - ``window_size`` = 7
        - ``embed_dim`` = 96
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (3, 6, 12, 24)

    To stay consistent with the architecture described in the paper, this class
    assumes the following:

        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
        drop_path_rate: The maximum probability of entirely skipping a (Shifted)
          Windows Multi-head Attention computation (Stochastic Depth
          computation-skipping technique) during training.
          This maximum value is used in the last stage of the network, while
          previous stages use linearly spaced values in the
          [0 ,``drop_path_rate``] interval.
    """

    def __init__(
        self,
        num_classes: int,
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )

        return config

    @classmethod
    def from_config(cls, config: dict) -> "SwinT":
        return Swin(**config)


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinB(Swin):
    """Swin-B Transformer Model.

    This version (base) uses the following options:

        - ``patch_size`` = 4
        - ``window_size`` = 7
        - ``embed_dim`` = 128
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (4, 8, 16, 32)

    To stay consistent with the architecture described in the paper, this class
    assumes the following:

        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
        drop_path_rate: The maximum probability of entirely skipping a (Shifted)
          Windows Multi-head Attention computation (Stochastic Depth
          computation-skipping technique) during training.
          This maximum value is used in the last stage of the network, while
          previous stages use linearly spaced values in the
          [0 ,``drop_path_rate``] interval.
    """

    def __init__(
        self,
        num_classes: int,
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            patch_size=4,
            window_size=7,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )

        return config

    @classmethod
    def from_config(cls, config: dict) -> "SwinT":
        return Swin(**config)


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinL(Swin):
    """Swin-L Transformer Model.

    This version (large) uses the following options:

        - ``patch_size`` = 4
        - ``window_size`` = 7
        - ``embed_dim`` = 192
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (6, 12, 24, 48)

    To stay consistent with the architecture described in the paper, this class
    assumes the following:

        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
        drop_path_rate: The maximum probability of entirely skipping a (Shifted)
          Windows Multi-head Attention computation (Stochastic Depth
          computation-skipping technique) during training.
          This maximum value is used in the last stage of the network, while
          previous stages use linearly spaced values in the
          [0 ,``drop_path_rate``] interval.
    """

    def __init__(
        self,
        num_classes: int,
        drop_rate: float = DEFAULT_DROP_RATE,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            patch_size=4,
            window_size=7,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )

        return config

    @classmethod
    def from_config(cls, config: dict) -> "SwinT":
        return Swin(**config)
