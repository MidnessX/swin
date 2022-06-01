"""The Swin model definition module."""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import collections.abc

from swin.modules import SwinPatchEmbeddings, SwinStage, SwinLinear


class Swin(tf.keras.Model):
    """Swin Transformer Model.

    Some assumptions have been made about this model:

        - ``inputs`` must always be a color image (3 channels).
        - The size of windows in (Shifted) Windows Multi-head Attention is fixed
          to 7.
        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to attention heads.
        - The probability of the Stochastic Depth computation-skipping technique
          during training is fixed to 0.1.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        inputs: The input to be expected by the model. It must describe a batch
            of images in the ``channels_last`` format. Images must have height
            equal to width (they must be square images).
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        patch_size: The size of each patch in which images will be divided into.
        embed_dim: The lenght of embeddings built from patches.
        depths: The number of ``SwinTransformer`` layers in each stage of the
            network.
        num_heads: The number of (Shifted) Windows Multi-head Attention heads in
            each stage of the network.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
    """

    def __init__(
        self,
        inputs: tf.Tensor,
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: collections.abc.Collection[int] = (2, 2, 6, 2),
        num_heads: collections.abc.Collection[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert inputs.dtype == tf.float32
        assert inputs.shape[1] == inputs.shape[2] and inputs.shape[3] == 3

        self.input_shape_list = [
            inputs.shape[0],
            inputs.shape[1],
            inputs.shape[2],
            inputs.shape[3],
        ]  # When returning this model's config, we only need axes' shapes, not the whole input tensor
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(self.depths)
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = 0.1

        self.patch_embed = SwinPatchEmbeddings(
            self.embed_dim,
            self.patch_size,
            norm_layer=True,
            name="patches_linear_embedding",
        )
        self.patch_embed.compute_output_shape(inputs.shape)

        self.pos_drop = tf.keras.layers.Dropout(rate=self.drop_rate)

        # We must use numpy to generate these values as using TensorFlow's
        # equivalent function would mean creating EagerTensor objects.
        # These tensor would then get returned as layer parameters through
        # calls to their get_config() methods, causing problems in the JSON
        # serialization as the built-in Python library cannot handle this
        # type of objects and thus preventing model saving.
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
                    input_resolution=self.patch_embed.patches_resolution[0] // (2**i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=7,
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
        self.avgpool = tfa.layers.AdaptiveAveragePooling1D(
            1, name="adaptive_average_pooling"
        )
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.head = SwinLinear(num_classes, name="classification_head")

    def call(self, inputs, **kwargs):
        x = self.patch_embed(inputs, **kwargs)
        x = self.pos_drop(x, **kwargs)
        x = self.blocks(x, **kwargs)
        x = self.norm(x, **kwargs)
        x = self.avgpool(x, **kwargs)
        x = self.flatten(x, **kwargs)

        x = self.head(x, **kwargs)
        x = tf.nn.softmax(x)

        return x

    def get_config(self) -> dict:
        config = {
            "input_shape_list": self.input_shape_list,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "drop_rate": self.drop_rate,
        }

        return config

    @classmethod
    def from_config(cls, config: dict) -> "Swin":
        # Since we only have the shape of the input, we build a new random tensor.
        # Dtype is fixed to tf.float32.
        inputs = tf.random.uniform(config.pop("input_shape_list"), dtype=tf.float32)

        return cls(inputs, **config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size}, embed_dim={self.embed_dim}, depths={self.depths}, num_heads={self.num_heads}, drop_rate={self.drop_rate})"


class SwinT(Swin):
    """Swin-T Transformer Model.

    This version (tiny) uses the following options:

        - ``patch_size`` = 4
        - ``embed_dim`` = 96
        - ``depths`` = (2, 2, 6, 2)
        - ``num_heads`` = (3, 6, 12, 24)

    Some assumptions have been made about this model:

        - ``inputs`` must always be a coloured image (3 channels)
        - The size of windows in (Shifted) Windows Multi-head Attention is fixed
          to 7.
        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to Attention heads.
        - The probability of the Stochastic Depth technique is fixed to 0.1.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        inputs: The input to be expected by the model. It must describe a batch
            of images in the ``channels_last`` format. Images must have height
            equal to width (they must be square images).
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
    """

    def __init__(
        self, inputs: tf.Tensor, num_classes: int, drop_rate: float = 0, **kwargs
    ) -> None:
        super().__init__(
            inputs,
            num_classes,
            patch_size=4,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            **kwargs,
        )


class SwinS(Swin):
    """Swin-S Transformer Model.

    This version (small) uses the following options:

        - ``patch_size`` = 4
        - ``embed_dim`` = 96
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (3, 6, 12, 24)

    Some assumptions have been made about this model:

        - ``inputs`` must always be a coloured image (3 channels)
        - The size of windows in (Shifted) Windows Multi-head Attention is fixed
          to 7.
        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to Attention heads.
        - The probability of the Stochastic Depth technique is fixed to 0.1.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        inputs: The input to be expected by the model. It must describe a batch
            of images in the ``channels_last`` format. Images must have height
            equal to width (they must be square images).
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
    """

    def __init__(
        self, inputs: tf.Tensor, num_classes: int, drop_rate: float = 0, **kwargs
    ) -> None:
        super().__init__(
            inputs,
            num_classes,
            patch_size=4,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=drop_rate,
            **kwargs,
        )


class SwinB(Swin):
    """Swin-B Transformer Model.

    This version (base) uses the following options:

        - ``patch_size`` = 4
        - ``embed_dim`` = 128
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (4, 8, 16, 32)

    Some assumptions have been made about this model:

        - ``inputs`` must always be a coloured image (3 channels)
        - The size of windows in (Shifted) Windows Multi-head Attention is fixed
          to 7.
        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to Attention heads.
        - The probability of the Stochastic Depth technique is fixed to 0.1.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        inputs: The input to be expected by the model. It must describe a batch
            of images in the ``channels_last`` format. Images must have height
            equal to width (they must be square images).
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
    """

    def __init__(
        self, inputs: tf.Tensor, num_classes: int, drop_rate: float = 0, **kwargs
    ) -> None:
        super().__init__(
            inputs,
            num_classes,
            patch_size=4,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            drop_rate=drop_rate,
            **kwargs,
        )


class SwinL(Swin):
    """Swin-L Transformer Model.

    This version (large) uses the following options:

        - ``patch_size`` = 4
        - ``embed_dim`` = 192
        - ``depths`` = (2, 2, 18, 2)
        - ``num_heads`` = (6, 12, 24, 48)

    Some assumptions have been made about this model:

        - ``inputs`` must always be a coloured image (3 channels)
        - The size of windows in (Shifted) Windows Multi-head Attention is fixed
          to 7.
        - The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed
          to 4.
        - A learnable bias is added to ``queries``, ``keys`` and ``values``
          when computing (Shifted) Window Multi-head Attention.
        - ``queries`` and ``keys`` are scaled by a factor of
          ``head_dimension**-0.5``.
        - No dropout is applied to Attention heads.
        - The probability of the Stochastic Depth technique is fixed to 0.1.
        - No absolute position information is included in embeddings.
        - ``LayerNormalization`` is applied after building patch embeddings.

    Args:
        inputs: The input to be expected by the model. It must describe a batch
            of images in the ``channels_last`` format. Images must have height
            equal to width (they must be square images).
        num_classes: The number of classes to predict. It determines the
            dimension of the output tensor.
        drop_rate: The probability of dropping connections in ``Dropout``
            layers.
    """

    def __init__(
        self, inputs: tf.Tensor, num_classes: int, drop_rate: float = 0, **kwargs
    ) -> None:
        super().__init__(
            inputs,
            num_classes,
            patch_size=4,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            drop_rate=drop_rate,
            **kwargs,
        )
