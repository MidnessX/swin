"""Modules used by the Swin Transformer."""

import collections.abc

import numpy as np
import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package=__package__)
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

    def __init__(self, units: int, use_bias: bool = True, **kwargs) -> None:
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


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinPatchEmbeddings(tf.keras.layers.Layer):
    """Patch embedding layer.

    This layer builds embeddings for every patch of an image.

    Args:
        embed_dim: Dimension of output embeddings.
        patch_size: Height/width of patches, expressed in pixels.
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

    def build(self, input_shape: tf.TensorShape | str) -> None:
        # We want a rank-4 tensor in the channels_last format.
        # The input image must be a square with a size divisibile by the
        # patch_size
        assert (
            len(input_shape) == 4
            and input_shape[3] == 3
            and input_shape[1] == input_shape[2]
            and input_shape[1] % self.patch_size == 0
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Build embeddings for every patch of the image.

        Args:
            inputs: A batch of images with shape ``(batch_size, height, width, channels)``.
                ``height`` and ``width`` must be identical.

        Returns:
            Embeddings, having shape ``(batch_size, height / patch_size,
            width / patch_size, embed_dim)``.
        """

        x = self.proj(inputs, **kwargs)

        if self.norm is not None:
            x = self.norm(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config().copy()
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


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinPatchMerging(tf.keras.layers.Layer):
    """Swin Patch Merging Layer.

    Args:
        input_resolution: The resolution of the original data, expressed in
            patches.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def build(self, input_shape: tf.TensorShape | str):
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2]
        assert input_shape[1] % 2 == 0

        self.reduction = SwinLinear(input_shape[3] * 2, use_bias=False)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Merge groups of 4 neighbouring of patches.

        This layer concatenates the features of groups of 4 neighbouring patches
        and project the concatenation into a space twice the length of the
        original feature space.

        Args:
            inputs: Tensor of patches, with shape
                ``(batch_size, height_patches, width_patches, embed_dim)`` with
                ``height_patches`` must be equal to ``width_patches``.

        Returns:
            Embeddings of merged patches, with shape ``(batch_size,
            heigth_patches /2, width_patches / 2, 2 * embed_dim)``.
        """

        x = tf.concat(
            [
                inputs[:, 0::2, 0::2, :],
                inputs[:, 1::2, 0::2, :],
                inputs[:, 0::2, 1::2, :],
                inputs[:, 1::2, 1::2, :],
            ],
            axis=-1,
        )  # [batch_size, height_patches / 2, width_patches / 2, 4 * embed_dim]

        x = self.norm(x, **kwargs)
        x = self.reduction(
            x, **kwargs
        )  # [batch_size, height_patches / 2, width_patches / 2, 2 * embed_dim]

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinStage(tf.keras.layers.Layer):
    """Stage of the Swin Network.

    Args:
        depth: Number of ``SwinTransformer`` layers in the stage.
        num_heads: Number of attention heads in each ``SwinTransformer`` layer.
        window_size: The size of window axes expressed in patches.
        mlp_ratio: The ratio between the size of the hidden layer and the size
            of the output layer in ``SwinMlp`` layers.
        drop_p: The probability of dropping connections in a ``SwinTransformer``
            layer during training.
        drop_path_p: The proabability of entirely skipping the computation of
            (Shifted) Windows Multi-head Self Attention during training
            (Stochastic Depth technique).
        downsample: Whether or not to apply downsampling through a
            ``SwinPatchMerging`` layer at the end of the stage.
    """

    def __init__(
        self,
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

        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_p = drop_p
        self.drop_path_p = drop_path_p
        self.downsample = downsample

        self.core = tf.keras.Sequential(
            [
                SwinTransformer(
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

        self.downsample_layer = SwinPatchMerging() if downsample else None

    def build(self, input_shape: tf.TensorShape | str):
        assert (
            len(input_shape) == 4
        )  # Must be batch_size, height_patches, width_patches, embed_dim
        assert input_shape[1] == input_shape[2]

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply transformations of the Swin stage to patches.

        Args:
            inputs: The input patches to the Swin stage, having shape
                ``(batch_size, height_patches, width_patches, embed_dim)``.
                ``height_patches`` must be equal to ``width_patches``.

        Returns:
            Transformed patches with shape ``(batch_size, height_patches / 2,
            width_patches / 2, embed_dim * 2)`` if ``downsample == True``
            or ``(batch_size, height_patches, width_patches, embed_dim)``
            if ``downsample == False``.
        """

        x = self.core(inputs, **kwargs)

        if self.downsample:
            x = self.downsample_layer(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "depth": self.depth,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_p": self.drop_p,
                "drop_path_p": self.drop_path_p,
                "downsample": self.downsample,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth={self.depth}, num_heads={self.num_heads}, window_size={self.window_size}, mlp_ratio={self.mlp_ratio}, drop_p={self.drop_p}, drop_path_p={self.drop_path_p}, downsample={self.downsample})"


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinWindowAttention(tf.keras.layers.Layer):
    """Swin (Shifted) Window Multi-head Self Attention Layer.

    Args:
        num_heads: The number of attention heads.
        proj_drop_r: The ratio of output weights that randomly get dropped
            during training.
    """

    def __init__(
        self,
        num_heads: int,
        proj_drop_r: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.proj_drop_r = proj_drop_r

        self.proj_drop = tf.keras.layers.Dropout(self.proj_drop_r)
        self.softmax = tf.keras.layers.Softmax(-1)

    @classmethod
    def build_relative_position_index(cls, window_size: int) -> tf.Tensor:
        """Build the table of relative position indices.

        This table is used as an index to the relative position table. For each
        pair of tokens in a window, this table allows to get the index in the
        relative position table.

        Args:
            window_size: The size of windows (expressed in patches) used during
                the (S)W-MSA.

        Returns:
            A ``Tensor`` with shape ``(window_size**2, window_size**2)``
            representing indices in the relative position table for each pair of
            patches in the window.
        """

        coords = tf.range(0, window_size)
        coords = tf.stack(tf.meshgrid(coords, coords, indexing="ij"))
        coords = tf.reshape(coords, [tf.shape(coords)[0], -1])

        rel_coords = tf.expand_dims(coords, 2) - tf.expand_dims(
            coords, 1
        )  # Make values relative
        rel_coords = tf.transpose(rel_coords, [1, 2, 0])

        rel_coords = tf.Variable(rel_coords)

        rel_coords[:, :, 0].assign(
            rel_coords[:, :, 0] + window_size - 1
        )  # Add offset to values
        rel_coords[:, :, 1].assign(rel_coords[:, :, 1] + window_size - 1)

        rel_coords[:, :, 0].assign(
            rel_coords[:, :, 0] * (2 * window_size - 1)
        )  # Shift values so indices for different patches do not share the same value

        rel_pos_index = tf.reduce_sum(rel_coords, -1)

        return rel_pos_index

    def build(self, input_shape: tf.TensorShape | str) -> None:
        assert len(input_shape) == 5
        assert input_shape[2] == input_shape[3]
        assert (
            input_shape[4] % self.num_heads == 0
        )  # embeddings dimension must be evenly divisible by the number of attention heads

        window_size = input_shape[2]
        embed_dim = input_shape[4]

        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # In the paper, sqrt(d)

        # The official implementation uses a custom function which defaults
        # to truncating the normal in the interval [-2, 2] while
        # tf.keras.initializers.TruncatedNormal() only allows values in the
        # interval [-stddev, +stddev]
        self.relative_position_bias_table = self.add_weight(
            shape=[
                (2 * window_size - 1) ** 2,
                self.num_heads,
            ],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            name="relative_position_bias_table",
            trainable=True,
        )

        self.relative_position_index = (
            tf.reshape(
                SwinWindowAttention.build_relative_position_index(window_size),
                [-1],
            ),
        )  # Flatten the matrix so it can be used to index the relative_position_bias_table in the forward pass

        self.qkv = SwinLinear(embed_dim * 3)
        self.proj = SwinLinear(embed_dim)

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor | None = None, **kwargs
    ) -> tf.Tensor:
        """Perform (Shifted) Window MSA.

        Args:
            inputs: Embeddings with shape ``(batch_size, num_windows,
                window_size, window_size, embed_dim)``. ``embed_dim`` must be
                exactly divisible by ``num_heads``.
            mask: Attention mask used used to perform Shifted Window MSA, having
                shape ``(num_windows, window_size * window_size, window_size * window_size)`` and values {0, -inf}.

        Returns:
            The result of (Shifted) Window MSA, having identical shape to the
            input.
        """

        shape = tf.shape(inputs)
        batch_windows = shape[0] * shape[1]
        window_dim = shape[2] * shape[3]
        embed_dim = shape[4]

        x = tf.reshape(inputs, [batch_windows, window_dim, embed_dim])

        qkv = self.qkv(x, **kwargs)  # [batch_windows, window_dim, 3 * embed_dim]
        qkv = tf.reshape(
            qkv,
            [batch_windows, window_dim, 3, self.num_heads, self.head_dim],
        )
        qkv = tf.transpose(
            qkv, [2, 0, 3, 1, 4]
        )  # [3, batch_windows, num_heads, window_dim, head_dim]

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = q * self.scale

        attn = tf.matmul(
            q, k, transpose_b=True
        )  # [batch_windows, num_heads, window_dim, window_dim]

        relative_position_bias = tf.gather(
            self.relative_position_bias_table, self.relative_position_index
        )  # [window_dim**2, num_heads]
        relative_position_bias = tf.reshape(
            relative_position_bias, [window_dim, window_dim, -1]
        )
        relative_position_bias = tf.transpose(
            relative_position_bias, [2, 0, 1]
        )  # [num_heads, window_dim, window_dim]

        attn = attn + tf.expand_dims(
            relative_position_bias, axis=0
        )  # [batch_windows, num_heads, window_dim, window_dim]

        if mask is not None:
            num_windows = tf.shape(mask)[0]
            attn = tf.reshape(
                attn,
                [
                    batch_windows // num_windows,
                    num_windows,
                    self.num_heads,
                    window_dim,
                    window_dim,
                ],
            )  # Expand to [batch_size, num_windows, num_heads, window_dim, windo_dim] in order to sum the attention mask
            attn = attn + tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(
                attn, [-1, self.num_heads, window_dim, window_dim]
            )  # Back to [batch_windows, num_heads, window_dim, window_dim]

        attn = self.softmax(attn, **kwargs)

        attn = tf.matmul(attn, v)
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, [batch_windows, window_dim, embed_dim])

        attn = self.proj(attn, **kwargs)
        attn = self.proj_drop(attn, **kwargs)

        attn = tf.reshape(attn, tf.shape(inputs))

        return attn

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "num_heads": self.num_heads,
                "proj_drop_r": self.proj_drop_r,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_heads={self.num_heads}, proj_drop_r={self.proj_drop_r})"


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinDropPath(tf.keras.layers.Layer):
    """Stochastic per-sample layer drop.

    This is an implementation of the stochastic depth technique described in the
    "Deep Networks with Stochastic Depth" paper by Huang et al.
    (https://arxiv.org/pdf/1603.09382.pdf).

    Examples in a batch have a probability to have their values set to 0.
    This is useful in conjunction with residual paths, as adding the residual
    connection with 0 yields the original example, as if other computations
    never took place in the main path.

    Args:
        drop_prob: The probability of entirely skipping the layer.
    """

    def __init__(self, drop_prob: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        assert drop_prob >= 0 and drop_prob <= 1

        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob

    def build(self, input_shape: tf.TensorShape | str) -> None:
        # We want to get a rank-1 tensor, with tf.rank(inputs) values all set to
        # 1 except for the first one, identical to the batch size.
        # e.g. [4, 1, 1, 1].
        self.shape = tf.ones([len(input_shape)], dtype=tf.int32)
        self.shape = tf.tensor_scatter_nd_update(self.shape, [[0]], [input_shape[0]])

    def call(
        self, inputs: tf.Tensor, training: tf.Tensor = None, **kwargs
    ) -> tf.Tensor:
        """Apply the stochastic depth technique.

        Args:
            inputs: The input data. The first dimension is assumed to be the
                ``batch_size``.
            training: Whether the forward pass is happening at training time
                or not. During inference (``training = False``) ``inputs`` is
                returned as-is (i.e. no drops).

        Returns:
            The input tensor with some values randomly set to 0.
        """

        if self.drop_prob == 0 or not training:
            return inputs

        rand_tensor = tf.constant(self.keep_prob, dtype=inputs.dtype)
        rand_tensor = rand_tensor + tf.random.uniform(
            self.shape, maxval=1.0, dtype=inputs.dtype
        )
        rand_tensor = tf.floor(rand_tensor)

        output = tf.divide(inputs, self.keep_prob) * rand_tensor

        return output

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update({"drop_prob": self.drop_prob})
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"


@tf.keras.saving.register_keras_serializable(package=__package__)
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

    def build(self, input_shape: tf.TensorShape | str) -> None:
        assert len(input_shape) == 4

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the transformations of the MLP.

        Args:
            inputs: The input data, having shape
            ``(batch_size, height_patches, width_patches, embed_size)``.

        Returns:
            The transformed inputs, with shape
            ``(batch_size, num_patches, out_features)``.
        """

        x = self.fc1(inputs, **kwargs)
        x = tf.nn.gelu(x)
        x = self.drop(x, **kwargs)
        x = self.fc2(x, **kwargs)
        x = self.drop(x, **kwargs)

        return x

    def get_config(self) -> dict:
        config = super().get_config().copy()
        config.update(
            {
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "drop_p": self.drop_p,
            }
        )
        return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_features={self.hidden_features}, out_features={self.out_features}, drop_p={self.drop_p})"


@tf.keras.saving.register_keras_serializable(package=__package__)
class SwinTransformer(tf.keras.layers.Layer):
    """Swin Transformer Layer.

    Args:
        num_heads: The number of (Shifted) Window Attention heads.
        window_size: The size of window axes, expressed in patches.
        shift_size: The value of shifting applied to windows, expressed in
            patches.
        mlp_ratio: The ratio between the size of the hidden layer and the
            size of the output layer in ``SwinMlp``.
        drop_p: The probability of dropping connections in ``Dropout`` layers
            during training.
        drop_path_p: The probability of entirely skipping a (Shifted) Windows
            Multi-head Self Attention computation during training.
    """

    def __init__(
        self,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: int,
        drop_p: float = 0.0,
        drop_path_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_p = drop_p
        self.drop_path_p = drop_path_p

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = SwinWindowAttention(self.num_heads, proj_drop_r=drop_p)
        # When drop_path_p == 0 SwinDropPath simply returns the same value
        self.drop_path = SwinDropPath(self.drop_path_p)

        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    @classmethod
    def window_partition(cls, patches: tf.Tensor, window_size: tf.Tensor) -> tf.Tensor:
        """Partition a batch of images into windows.

        Args:
            patches: A batch of patch embeddings to partition into windows,
                having shape ``(batch_size, num_patches_h, num_patches_w,
                embed_dim)``.
            window_size: The size of each window, expressed in patches along
                each axis.

        Returns:
            A tensor of windows having shape ``(batch_size, n, window_size,
            window_size, embed_dim)``, where ``n`` is the number of
            resulting windows.
        """

        shape = tf.shape(patches)

        windows = tf.reshape(
            patches,
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
        windows = tf.reshape(
            windows, [shape[0], -1, window_size, window_size, shape[3]]
        )

        return windows

    @classmethod
    def window_reverse(cls, windows: tf.Tensor, resolution: tf.Tensor) -> tf.Tensor:
        """Reverse the partitioning of a batch of patches into windows.

        .. Note:
            ``resolution`` is expected to be a multiple of the size of windows.
            No checks are performed to ensure this holds.

        Args:
            windows: Partitioned windows to reverse, with shape
            ``(batch_size, num_windows, window_size, window_size, embed_dim)``.
            resolution: Number of patches per axis in the original feature map.

        Returns:
            A tensor of patches of the batch recreated from ``windows``, with
            shape ``(batch_size, resolution, resolution, embed_dim)``.
        """

        shape = tf.shape(windows)

        batch_size = shape[0]
        window_size = shape[2]
        embed_dim = shape[4]

        x = tf.reshape(
            windows,
            [
                batch_size,
                resolution // window_size,
                resolution // window_size,
                window_size,
                window_size,
                embed_dim,
            ],
        )
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [batch_size, resolution, resolution, embed_dim])

        return x

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

        # While possible to build the mask only through TensorFlow operations,
        # it would result in a much less readable method.Since Numpy is already
        # a TensorFlow dependency and this method is only called during this
        # layer's initialization, using it to build the mask is fine.
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
        )  # mask_windows.shape = [n, window_size, window_size, 1].
        mask_windows = tf.reshape(
            mask_windows, [-1, window_size * window_size]
        )  # mask_windows.shape = [n, window_size**2], we flatten windows.

        # We need to create a mask which, for each patch in each window, tells
        # us if the attention mechanism should be calculated for every other
        # patch in the same window.
        # This means a mask with shape [n, window_size**2, window_size**2].
        # Subtracting the two expanded mask_windows gives us a tensor with the
        # right shape and values equal to zero where two patches are adjacent in
        # the original feature map (meaning attention should be calculated).
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)

        # We now need to change values != 0 to something negative. When put
        # through the SoftMax operation performed during the SW-MSA, it results
        # in a value close to 0 for those patches that were not adjacent in the
        # original feature map.
        # Technically, the bigger the negative number the better
        # (i.e. -math.inf), but it could lead to float values shenanigans so we
        # choose -100 to stay consistent with the original implementation.
        attn_mask = tf.where(attn_mask != 0, tf.constant(-100.0), attn_mask)

        return attn_mask

    def build(self, input_shape: tf.TensorShape | str) -> None:
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2]

        self.resolution = input_shape[1]

        # Resolution must be evenly divisible by the window size or reshape
        # operations will not work
        assert self.resolution % self.window_size == 0

        if self.resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.resolution

        assert 0 <= self.shift_size < self.window_size

        if self.shift_size > 0:
            self.attn_mask = self.build_attn_mask(
                self.resolution,
                self.window_size,
                self.shift_size,
            )
        else:
            self.attn_mask = None

        dim = input_shape[-1]
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = SwinMlp(mlp_hidden_dim, out_features=dim, drop_p=self.drop_p)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """Apply the transformations of the transformer layer.

        Args:
            inputs: Input embeddings with shape
            ``(batch_size, height_patches, width_patches, embed_dim)``.
            ``height_patches`` must be equal to ``width_patches``.

        Returns:
            Transformed embeddings with same shape as ``inputs``.
        """

        shortcut_1 = inputs

        # Layer normalization
        x = self.norm_1(inputs, **kwargs)

        # Cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        # Window partitioning
        x = self.window_partition(x, self.window_size)

        # (Shifted) Window Multi-head Self Attention
        x = self.attention(x, mask=self.attn_mask, **kwargs)

        # Undo window partitioning (window merging)
        x = self.window_reverse(x, tf.constant(self.resolution))

        # Undo cyclic shift (reverse cyclic shift)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

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
        config = super().get_config().copy()
        config.update(
            {
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
        return f"{self.__class__.__name__}(window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, drop_p={self.drop_p}, drop_path_p={self.drop_path_p})"
