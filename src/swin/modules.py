"""Modules used by the Swin Transformer."""

import collections.abc
import numpy as np
import tensorflow as tf


class SwinLinear(tf.keras.layers.Dense):
    """Linear layer.

    This layer applies a linear transformation to inputs and initialises its
    weights according to a truncated normal distribution with mean 0 and
    standard deviation 0.02.
    Its biases, if present, are initialised to 0.

    Args:
        units: The number of output units.
        use_bias: Whether the layer uses a bias vector.
    """

    def __init__(self, units: int, use_bias=True, **kwargs) -> None:
        super().__init__(
            units,
            activation=tf.keras.activations.linear,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            bias_initializer=tf.keras.initializers.Constant(0),
            **kwargs,
        )


class SwinPatchEmbeddings(tf.keras.layers.Layer):
    """Patch embedding layer.

    This layer builds embeddings for every patch of an image.

    Args:
        embed_dim: Dimension of output embeddings.
        patch_size: Size of axes of image patches, expressed in pixels.
        norm_layer: Whether to apply layer normalization or not.
    """

    def __init__(
        self, embed_dim: int, patch_size: int, norm_layer: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm_layer = norm_layer

        self.proj = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
        )
        self.norm = (
            tf.keras.layers.LayerNormalization(epsilon=1e-5)
            if self.norm_layer
            else None
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        # We want a rank-4 tensor in the channels_last format.
        # The input image must be a square with a size divisibile by the
        # patch_size
        assert (
            input_shape.rank == 4
            and input_shape[3] == 3
            and input_shape[1] == input_shape[2]
            and input_shape[1] % self.patch_size == 0
        )

        self.patches_resolution = (
            input_shape[1] // self.patch_size,
            input_shape[2] // self.patch_size,
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.flatten = tf.keras.layers.Reshape((-1, self.embed_dim))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Build embeddings for every patch of the image.

        Args:
            inputs: A batch of images with shape (batch_size, height, width,
                channels).

        Returns:
            Embeddings, having shape ``(batch_size, num_patches, embed_dim)``.
        """

        x = tf.ensure_shape(inputs, [None, None, None, 3])

        x = self.proj(x, **kwargs)
        x = self.flatten(x, **kwargs)

        if self.norm:
            x = self.norm(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "patch_size": self.patch_size,
                "norm_layer": self.norm_layer,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim}, patch_size={self.patch_size}, norm_layer={self.norm_layer})"


class SwinPatchMerging(tf.keras.layers.Layer):
    """Swin Patch Merging Layer.

    Args:
        input_resolution: The resolution of the original data, expressed in
            patches.
    """

    def __init__(self, input_resolution: int, **kwargs) -> None:
        # NOTE: Changed input_resolution from tuple to int

        super().__init__(**kwargs)

        assert input_resolution % 2 == 0
        self.input_resolution = input_resolution

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def build(self, input_shape: tf.TensorShape):
        self.reduction = SwinLinear(input_shape[-1] * 2, use_bias=False)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Perform the merging of patches.

        The merge is performed on groups of 4 neighbouring patches.

        Args:
            inputs: Tensor of patches, with shape ``(batch_size,
                num_patches, embed_dim)`` with
                ``num_patches = input_resolution * input_resolution``.

        Returns:
            Embeddings of merged patches, with shape ``(batch_size, num_patches / 4, 2 * embed_dim)``.
        """

        tf.assert_equal(inputs.dtype, tf.float32, "Inputs must be a tf.float32 tensor.")
        x = tf.ensure_shape(inputs, [None, self.input_resolution**2, None])

        shape = tf.shape(inputs)
        batch = shape[0]
        channels = shape[2]

        x = tf.reshape(
            x, [batch, self.input_resolution, self.input_resolution, channels]
        )

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, [batch, -1, 4 * channels])

        x = self.norm(x, **kwargs)
        x = self.reduction(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"input_resolution": self.input_resolution})
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_resolution={self.input_resolution})"


class SwinStage(tf.keras.layers.Layer):
    """Stage of the Swin Network.

    Args:
        input_resolution: The resolution of axes of the input, expressed in
            number of patches.
        depth: Number of SwinTransformer layers in the stage.
        num_heads: Number of attention heads in each SwinTransformer layer.
        window_size: The size of windows in which embeddings gets split into,
            expressed in numer of patches.
        mlp_ratio: The ratio between the size of the hidden layer and the size
            of the output layer in SwinMlp layers.
        drop_p: The probability of dropping connections in a SwinTransformer
            layer during training.
        drop_path_p: The proabability of entirely skipping the computation of
            (Shifted) Windows Multi-head Self Attention during training
            (Stochastic Depth technique).
        downsample: Whether or not to apply downsampling at the end of the
            layer.
    """

    def __init__(
        self,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop_p: float = 0.0,
        drop_path_p: float | collections.abc.Sequence[float] = 0.0,
        downsample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_p = drop_p
        self.drop_path_p = drop_path_p
        self.donwsample = downsample

        self.core = tf.keras.Sequential(
            [
                SwinTransformer(
                    resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop_p=drop_p,
                    drop_path_p=drop_path_p[i]
                    if isinstance(drop_path_p, collections.abc.Sequence)
                    else drop_path_p,
                )
                for i in range(depth)
            ]
        )

        if downsample:
            self.downsample_layer = SwinPatchMerging(self.input_resolution)
        else:
            self.downsample_layer = None

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply transformations of the Swin stage to patches.

        Args:
            inputs: The input patches to the Swin stage, having shape ``
                (batch_size, num_patches, embed_dim)``.

        Returns:
            Transformed patches with shape ``(batch_size, num_patches / 4,
            embed_dim * 2)`` if ``downsample == True`` or ``(batch_size,
            num_patches, embed_dim)`` if ``downsample == False``.
        """

        x = tf.ensure_shape(inputs, [None, None, None])

        x = self.core(x, **kwargs)

        if self.donwsample:
            x = self.downsample_layer(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_p": self.drop_p,
                "drop_path_p": self.drop_path_p,
                "downsample": self.donwsample,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_resolution={self.input_resolution}, depth={self.depth}, num_heads={self.num_heads}, window_size={self.window_size}, mlp_ratio={self.mlp_ratio}, drop_p={self.drop_p}, drop_path_p={self.drop_path_p}, downsample={self.donwsample})"


class SwinWindowAttention(tf.keras.layers.Layer):
    """Swin (Shifted) Window Multi-head Self Attention Layer.

    Args:
        window_size: The size of windows in which embeddings gets divided into,
            expressed in patches.
        num_heads: The number of attention heads.
        proj_drop_r: The ratio of output weights that randomly get dropped
            during training.
    """

    def __init__(
        self,
        window_size: int,
        num_heads: int,
        proj_drop_r: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.window_size = window_size
        self.num_heads = num_heads
        self.proj_drop_r = proj_drop_r

        # TODO: Change into TF calls to get rid of numpy
        coords_h = range(self.window_size)
        coords_w = range(self.window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = np.reshape(coords, [coords.shape[0], -1])
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = np.transpose(relative_coords, [1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name="relative_position_index",
        )

        self.proj_drop = tf.keras.layers.Dropout(self.proj_drop_r)
        self.softmax = tf.keras.layers.Softmax(-1)

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = input_shape[-1]

        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim**-0.5  # In the paper, sqrt(d)

        # The official implementation uses a custom function which defaults
        # to truncating the normal in the interval [-2, 2] while
        # tf.keras.initializers.TruncatedNormal() only allows values in the
        # interval [-stddev, +stddev]
        self.relative_position_bias_table = self.add_weight(
            shape=[
                (2 * self.window_size - 1) ** 2,
                self.num_heads,
            ],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            name="relative_position_bias_table",
            trainable=True,
        )

        self.qkv = SwinLinear(channels * 3)
        self.proj = SwinLinear(channels)

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor | None = None, **kwargs
    ) -> tf.Tensor:
        """Perform (Shifted) Window MSA.

        Args:
            inputs: Embeddings with shape ``(num_windows * batch_size,
                window_size * window_size, embed_dim)``. ``embed_dim`` must be
                exactly divisible by ``num_heads``.
            mask: Attention mask used used to perform Shifted Window MSA, having
                shape ``(num_windows, window_size * window_size, window_size * window_size)`` and values {0, -inf}.

        Returns:
            The result of (Shifted) Window MSA, having identical shape to the
            input.
        """

        x = tf.ensure_shape(inputs, [None, self.window_size**2, None])

        shape = tf.shape(inputs)
        batch_windows = shape[0]
        window_dim = shape[1]
        embed_dim = shape[2]

        tf.assert_equal(
            embed_dim % self.num_heads,
            0,
            "Provided input dimension 3 (embed_dim) is not evenly divisible by the number of attention heads.",
        )

        qkv = self.qkv(x, **kwargs)
        qkv = tf.reshape(
            qkv,
            [batch_windows, window_dim, 3, self.num_heads, embed_dim // self.num_heads],
        )
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = q * self.scale

        attn = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))

        indices = tf.reshape(self.relative_position_index, [-1])
        relative_position_bias = tf.gather(self.relative_position_bias_table, indices)
        relative_position_bias = tf.reshape(
            relative_position_bias, [window_dim, window_dim, -1]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])

        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, [batch_windows // nW, nW, self.num_heads, window_dim, window_dim]
            )
            attn = attn + tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(attn, [-1, self.num_heads, window_dim, window_dim])

        attn = self.softmax(attn, **kwargs)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_windows, window_dim, embed_dim])
        x = self.proj(x, **kwargs)
        x = self.proj_drop(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "proj_drop_r": self.proj_drop_r,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_size={self.window_size}, num_heads={self.num_heads}, proj_drop_r={self.proj_drop_r})"


class SwinDropPath(tf.keras.layers.Layer):
    """Stochastic Depth Layer.

    Args:
        drop_prob: The probability of entirely skipping the output of the
            computation.
    """

    def __init__(self, drop_prob: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob

    def call(
        self, inputs: tf.Tensor, training: tf.Tensor = None, **kwargs
    ) -> tf.Tensor:
        """Apply the stochastic depth technique.

        Args:
            inputs: The input data. The first dimension is assumed to be the
                ``batch_size``.
            training: Whether the forward pass is happening at training time
                or not. During inference (``training`` = False) ``inputs`` is
                returned as-is.

        Returns:
            The input tensor with some values randomly set to 0.
        """

        if self.drop_prob == 0 or not training:
            return inputs

        first_axis = tf.expand_dims(tf.shape(inputs)[0], axis=0)
        other_axis = tf.repeat(
            1, tf.rank(inputs) - 1
        )  # Rank-1 tensor with (rank(inputs) - 1) axes, all having value 1

        # We want to get a rank-1 tensor with 1 as the value of all axes except
        # for the first one, identical to the batch size
        shape = tf.concat(
            [first_axis, other_axis],
            axis=0,
        )

        rand_tensor = tf.constant(self.keep_prob, dtype=inputs.dtype)
        rand_tensor = rand_tensor + tf.random.uniform(
            shape, maxval=1.0, dtype=inputs.dtype
        )
        rand_tensor = tf.floor(rand_tensor)

        output = tf.divide(inputs, self.keep_prob) * rand_tensor

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"


class SwinMlp(tf.keras.layers.Layer):
    """Swin Multi-layer Perceptron Layer.

    Args:
        hidden_features: The number of features in the hidden layer.
        out_features: The number of output features of this layer.
        drop_p: The probability of dropping input signals during training.
    """

    def __init__(
        self, hidden_features: int, out_features: int, drop_p: float = 0.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop_p = drop_p

        self.fc1 = SwinLinear(self.hidden_features)
        self.fc2 = SwinLinear(self.out_features)
        self.drop = tf.keras.layers.Dropout(self.drop_p)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the transformations of the MLP.

        Args:
            inputs: The input data, having shape ``(batch_size, num_patches,
                embed_size)``.

        Returns:
            The transformed inputs, with shape ``(batch_size, num_patches,
            out_features)``.
        """
        x = tf.ensure_shape(inputs, [None, None, None])

        x = self.fc1(x, **kwargs)
        x = tf.nn.gelu(x)
        x = self.drop(x, **kwargs)
        x = self.fc2(x, **kwargs)
        x = self.drop(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_features={self.hidden_features}, out_features={self.out_features}, drop_p={self.drop_p})"


class SwinTransformer(tf.keras.layers.Layer):
    """Swin Transformer Layer.

    Args:
        resolution: The input resolution expressed in number of patches per
            axis. Both axis share the same resolution as the orginal image
            must be a square.
        num_heads: The number of Shifted Window Attention heads.
        window_size: The size of windows in which the image gets partitioned
            into, expressed in patches.
        shift_size: The value of shifting applied to windows, expressed in
            patches.
        mlp_ratio: The ratio between the size of the hidden layer and the
            size of the output layer in SwinMlp.
        drop_p: The probability of dropping connections in Dropout layers during
            training.
        drop_path_p: The probability of entirely skipping a (Shifted) Windows
            Multi-head Self Attention computation during training.
    """

    def __init__(
        self,
        resolution: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: int,
        drop_p: float = 0.0,
        drop_path_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.resolution = resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_p = drop_p
        self.drop_path_p = drop_path_p

        if self.resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.resolution

        # Resolution must be evenly divisible by the window size or reshape
        # operations will not work
        assert self.resolution % self.window_size == 0

        assert 0 <= self.shift_size < self.window_size

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = SwinWindowAttention(
            self.window_size, self.num_heads, proj_drop_r=drop_p
        )
        # When drop_path_p == 0 SwinDropPath simply returns the same value
        self.drop_path = SwinDropPath(self.drop_path_p)

        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        if self.shift_size > 0:
            attn_mask = self.build_attn_mask(
                self.resolution,
                self.window_size,
                self.shift_size,
            )

            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name="attention_mask",
            )
        else:
            self.attn_mask = None

    @classmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ],
    )
    def window_partition(cls, patches: tf.Tensor, window_size: tf.Tensor) -> tf.Tensor:
        """Partition a batch of images into windows.

        .. Note::

            This method may throw warnings due to an excessive number of
            retracing operations.
            However, due to it being used in the forward pass of the full
            model, keeping it decorated as a ``tf.function`` should still prove
            to be beneficial.

        Args:
            patches: Patch embeddings for a batch of images to partition
                into windows, having shape ``(batch_size, num_patches_h,
                num_patches_w, embed_dim)``. ``num_patches_h == num_patches_w``.
            window_size: The size of each window, expressed in patches.

        Returns:
            A tensor of windows having shape ``(n * batch_size, window_size,
            window_size, embed_dim)``, where ``n`` is the number of
            resulting windows.
        """

        x = tf.ensure_shape(patches, [None, None, None, None])
        window_size = tf.ensure_shape(window_size, [])

        shape = tf.shape(x)
        tf.assert_equal(
            shape[1],
            shape[2],
            "The number of patches in the height dimension must be equal to the number of patches in the width dimension (patches must be squared).",
        )

        windows = tf.reshape(
            x,
            [
                shape[0],
                shape[1] // window_size,
                window_size,
                shape[2] // window_size,
                window_size,
                shape[3],
            ],
        )
        windows = tf.transpose(windows, [0, 1, 3, 2, 4, 5])
        windows = tf.reshape(windows, [-1, window_size, window_size, shape[3]])

        return windows

    @classmethod
    @tf.function
    def window_reverse(cls, windows: tf.Tensor, patch_size: tf.Tensor) -> tf.Tensor:
        """Reverse the partitioning of a batch of patches into windows.

        Args:
            windows: Partitioned windows to reverse, with shape ``(batch_size *
                num_windows, window_size, window_size, embed_dim)``.
            patch_size: Number of patches per axis in the original image.

        Returns:
            A tensor of patches of the batch recreated from ``windows``, with
            shape ``(batch_size, patch_size, patch_size, embed_dim)``.
        """

        x = tf.ensure_shape(windows, [None, None, None, None])

        tf.assert_equal(
            tf.shape(x)[1],
            tf.shape(x)[2],
            "Dimension 1 and dimension 2 of 'windows' must be identical.",
        )
        window_size = tf.shape(x)[1]

        # TODO: simplify
        b = tf.cast(tf.shape(x)[0], tf.float64)  # Casting to prevent type mismatch
        d = patch_size**2 / window_size / tf.cast(window_size, tf.float64)
        batch_size = tf.cast(b / d, tf.int32)

        x = tf.reshape(
            x,
            [
                batch_size,
                patch_size // window_size,
                patch_size // window_size,
                window_size,
                window_size,
                -1,
            ],
        )
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [batch_size, patch_size, patch_size, -1])

        return x

    @classmethod
    def masked_fill(
        cls, tensor: tf.Tensor, mask: tf.Tensor, value: tf.Tensor
    ) -> tf.Tensor:
        """Fill elements of ``tensor`` with ``value`` where ``mask`` is True.

        This function returns a new tensor having the same values as ``tensor``
        except for those where ``mask`` contained the value True; these values are
        replaced with ``value``.

        It mimics ``torch.tensor.masked_fill()``.

        ``mask`` must have identical shape to ``tensor`` and ``value`` must be a
        scalar tensor.
        ``value`` is cast to the type of ``tensor`` if their types don't match.

        Args:
            tensor: The tensor to fill with ``value`` where ``mask`` is True.
            mask: The mask to apply to ``tensor``.
            value: The value to fill ``tensor`` with.

        Returns:
            A copy of ``tensor`` with elements changed to ``value`` where
            ``mask`` was ``True``.
        """

        tf.assert_equal(
            tf.shape(tensor),
            tf.shape(mask),
            "The shape of tensor must match the shape of mask.",
        )
        tf.assert_equal(tf.rank(value), 0, "'value' must be a scalar tensor.")

        if value.dtype != tensor.dtype:
            value = tf.cast(value, tensor.dtype)

        indices = tf.where(mask)

        filled_tensor = tf.tensor_scatter_nd_update(
            tensor,
            indices,
            tf.broadcast_to(
                value,
                [tf.shape(indices)[0]],
            ),
        )

        return filled_tensor

    @classmethod
    def build_attn_mask(cls, size: int, window_size: int, shift_size: int):
        """Build an attention mask for the Shifted Window MSA.

        Args:
            size: The number of patches per axis.
            window_size: The size of windows, expressed in patches.
            shift_size: The shifting applied to windows, expressed in patches.

        Returns:
            The computed attention mask, with shape ``(num_windows, window_size * window_size, window_size * window_size)``.
        """

        # TODO: Change mask creation to ditch numpy
        mask = np.zeros(
            [1, size, size, 1], dtype=np.float32
        )  # Force type so we get a tf.float32 tensor as the output of this method.
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )

        i = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice, w_slice, :] = i
                i += 1

        mask_windows = SwinTransformer.window_partition(
            tf.convert_to_tensor(mask), tf.constant(window_size)
        )
        mask_windows = tf.reshape(mask_windows, [-1, window_size * window_size])
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = SwinTransformer.masked_fill(
            attn_mask, attn_mask != 0, tf.constant(-100.0)
        )  # TODO: check if -100 can be changed to -math.inf
        attn_mask = SwinTransformer.masked_fill(
            attn_mask, attn_mask == 0, tf.constant(0.0)
        )

        return attn_mask

    def build(self, input_shape: tf.TensorShape) -> None:
        dim = input_shape[-1]
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = SwinMlp(mlp_hidden_dim, out_features=dim, drop_p=self.drop_p)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the transformations of the transformer layer.

        Args:
            inputs: Input embeddings with shape ``(batch_size, num_patches, embed_dim)``.

        Returns:
            Transformed embeddings with same shape as ``inputs``.
        """

        x = tf.ensure_shape(inputs, [None, self.resolution * self.resolution, None])

        shape = tf.shape(inputs)

        batch = shape[0]
        channels = shape[2]

        shortcut_1 = x

        x = self.norm_1(x, **kwargs)
        x = tf.reshape(x, [batch, self.resolution, self.resolution, channels])
        shifted_x = x

        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )

        # Window partitioning
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, [-1, self.window_size * self.window_size, channels]
        )

        # (Shifted) Window Multi-head Self Attention
        attn_windows = self.attention(x_windows, mask=self.attn_mask, **kwargs)

        # Window merging
        attn_windows = tf.reshape(
            attn_windows, [-1, self.window_size, self.window_size, channels]
        )
        shifted_x = self.window_reverse(
            attn_windows,
            tf.constant(self.resolution),
        )

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, [batch, self.resolution * self.resolution, channels])

        # Sum the skip connection and the output of (S)W-MSA
        x = shortcut_1 + self.drop_path(x, **kwargs)

        # Feed-forward network
        shortcut_2 = x
        x = self.norm_2(x, **kwargs)
        x = self.mlp(x, **kwargs)

        # Sum the skip connection and the output of the FFN
        x = shortcut_2 + self.drop_path(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "resolution": self.resolution,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_p": self.drop_p,
                "drop_path_p": self.drop_path_p,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(resolution={self.resolution}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, drop_p={self.drop_p}, drop_path_p={self.drop_path_p})"
