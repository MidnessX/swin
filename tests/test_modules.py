"""Module containing tests for the modules of the Swin Transformer network."""

import tensorflow as tf
import random
import unittest
import swin.modules as sm


class TestSwinLinear(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = random.randint(1, 5)
        self.img_size = 2 ** random.randint(5, 10)  # 32-1024 px

        self.input = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, 3], dtype=tf.float32
        )

        self.units = random.randint(1, 5)
        self.layer = sm.SwinLinear(self.units)

    def test_output(self) -> None:
        output = self.layer(self.input)
        shape = output.shape
        self.assertEqual(shape[0], self.batch_size)
        self.assertEqual(shape[-1], self.units)
        self.assertEqual(output.dtype, tf.float32)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 2)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestSwinPatchEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = random.randint(1, 5)
        self.img_size = 2 ** random.randint(5, 10)  # 32-1024 px

        self.input = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, 3], dtype=tf.float32
        )
        self.wrong_input_shape = tf.random.uniform(
            [self.batch_size, self.img_size, 2 * self.img_size, 3], dtype=tf.float32
        )
        self.wrong_input_dtype = tf.image.convert_image_dtype(
            tf.random.uniform(
                [self.batch_size, self.img_size, self.img_size, 3],
                maxval=255,
                dtype=tf.float32,
            ),
            tf.uint8,
        )
        self.wrong_input_channels = self.wrong_input_shape = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, 1], dtype=tf.float32
        )

        self.wrong_inputs = [
            self.wrong_input_shape,
            self.wrong_input_channels,
            self.wrong_input_dtype,
        ]

        self.embed_dim = 2 ** random.randint(5, 8)  # 32-256
        self.patch_size = 2 ** random.randint(
            1, 5
        )  # 2-32 px, never exceed max img_size
        self.layer = sm.SwinPatchEmbeddings(self.embed_dim, self.patch_size)

    def test_output(self) -> None:
        output = self.layer(self.input)

        shape = output.shape

        self.assertEqual(shape[0], self.batch_size)
        self.assertEqual(shape[-1], self.embed_dim)
        self.assertEqual(output.dtype, tf.float32)

    def test_wrong_input(self) -> None:
        for input_data in self.wrong_inputs:
            with self.subTest(input_data):
                self.assertRaises(Exception, self.layer, input_data)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 4)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestSwinPatchMerging(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = random.randint(1, 5)
        self.patch_size = 2 * random.randint(7, 28)  # Any even number would be fine
        self.embed_dim = 2 ** random.randint(5, 8)  # 32-256

        self.input = tf.random.uniform(
            [self.batch_size, self.patch_size**2, self.embed_dim], dtype=tf.float32
        )

        self.layer = sm.SwinPatchMerging(self.patch_size)

    def test_build_odd_patch_size(self) -> None:
        self.assertRaises(Exception, sm.SwinPatchMerging, 7)  # Any odd number is fine

    def test_output(self) -> None:
        output = self.layer(self.input)

        shape = output.shape

        self.assertEqual(len(shape), 3)
        self.assertEqual(shape[0], self.batch_size)
        self.assertEqual(shape[1], self.patch_size**2 / 4)
        self.assertEqual(shape[2], self.embed_dim * 2)
        self.assertEqual(output.dtype, tf.float32)

    def test_wrong_input(self) -> None:
        self.wrong_input_shape = tf.random.uniform(
            [self.batch_size, self.patch_size * (self.patch_size * 2), self.embed_dim],
            dtype=tf.float32,
        )
        self.wrong_input_dtype = tf.random.uniform(
            [self.batch_size, self.patch_size**2, self.embed_dim],
            maxval=255,
            dtype=tf.int32,
        )

        self.wrong_inputs = [
            self.wrong_input_shape,
            self.wrong_input_dtype,
        ]

        for input_data in self.wrong_inputs:
            with self.subTest(f"Shape: {input_data.shape}, dtype: {input_data.dtype}"):
                self.assertRaises(Exception, self.layer, input_data)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 3)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestSwinDropPath(unittest.TestCase):
    def setUp(self) -> None:
        self.input = tf.random.uniform(
            [4, 224, 224, 3], dtype=tf.float32
        )  # Any shape and dtype would be ok

    def test_output_dtype(self) -> None:
        layer = sm.SwinDropPath(0.5)

        output = layer(self.input, training=True)

        self.assertEqual(output.dtype, tf.float32)

    def test_output_shape(self) -> None:
        layer = sm.SwinDropPath(0.5)

        output = layer(self.input, training=True)

        self.assertEqual(self.input.shape, output.shape)

        output = layer(self.input, training=False)

        self.assertEqual(self.input.shape, output.shape)

    def test_drop_prob_zero(self) -> None:
        layer = sm.SwinDropPath(0)

        output = layer(self.input, training=True)

        self.assertTrue(tf.reduce_all(self.input == output))

    def test_drop_prob_not_zero(self) -> None:
        layer = sm.SwinDropPath(0.1)

        output = layer(self.input, training=True)

        self.assertFalse(tf.reduce_all(self.input == output))

    def test_not_training(self) -> None:
        layer = sm.SwinDropPath(0.1)

        output = layer(self.input, training=False)

        self.assertTrue(tf.reduce_all(self.input == output))


class TestSwinMlp(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_features = random.randint(1, 256)
        self.out_features = random.randint(1, 256)
        self.drop_p = random.random()

        self.layer = sm.SwinMlp(self.hidden_features, self.out_features, self.drop_p)

        self.batch_size = random.randint(1, 5)
        self.input = tf.random.uniform([self.batch_size, 768, 96], dtype=tf.float32)

    def test_output(self) -> None:
        output = self.layer(self.input)
        shape = output.shape

        self.assertEqual(shape[0], self.batch_size)
        self.assertEqual(shape[-1], self.out_features)
        self.assertEqual(output.dtype, tf.float32)

    def test_wrong_input(self) -> None:
        wrong_input = tf.random.uniform([self.batch_size, 224, 224, 96])

        self.assertRaises(Exception, self.layer, wrong_input)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 4)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestWindowAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.window_size = 2 * random.randint(
            1, 5
        )  # Could also be odd, but it simplifies the following operations
        self.num_heads = random.randint(1, 5)
        self.proj_drop_r = random.random()

        self.layer = sm.SwinWindowAttention(
            self.window_size, self.num_heads, self.proj_drop_r
        )

        self.batch_size = random.randint(1, 5)
        self.embed_dim = self.num_heads * random.randint(
            5, 10
        )  # Any value divisible by num_heads would be fine
        self.num_patches = self.window_size * random.randint(
            2, 10
        )  # Must be divisible by window_size
        self.num_windows = int((self.num_patches / self.window_size)) ** 2

        self.input = tf.random.uniform(
            [
                self.batch_size * self.num_windows,
                self.window_size**2,
                self.embed_dim,
            ],
            dtype=tf.float32,
        )

        self.attention_mask = sm.SwinTransformer.build_attn_mask(
            self.num_patches, self.window_size, self.window_size // 2
        )

    def test_output_no_shift(self) -> None:
        output = self.layer(self.input)

        self.assertEqual(output.shape, self.input.shape)
        self.assertEqual(output.dtype, self.input.dtype)

    def test_output_shift(self) -> None:
        output = self.layer(self.input, self.attention_mask)

        self.assertEqual(output.shape, self.input.shape)
        self.assertEqual(output.dtype, self.input.dtype)

    def test_wrong_input(self) -> None:
        wrong_inputs = []
        wrong_inputs.append(
            tf.random.uniform(
                [
                    self.batch_size * self.num_windows,
                    self.window_size**2 - 1,
                    self.embed_dim,
                ],
                dtype=tf.float32,
            )
        )  # Wrong shape
        wrong_inputs.append(
            tf.random.uniform(
                [
                    self.batch_size * self.num_windows,
                    self.window_size**2,
                    self.num_heads * 2 + 1,
                ],
                dtype=tf.float32,
            )
        )  # Incompatible emebedding dimensions

        for input_data in wrong_inputs:
            with self.subTest(input_data):
                self.assertRaises(Exception, self.layer, input_data)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 5)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestSwinTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.resolution = 2 ** random.randint(3, 6)  # 8-64
        self.num_heads = random.randint(1, 4)
        # Window size must be evenly divisible by resolution and > 0. We choose
        # > 1 to simplify tests where having window_size = resolution would
        # require a lot more code
        self.window_size = int(self.resolution / (2 ** random.randint(1, 2)))
        self.shift_size = int(self.window_size / 2)
        self.mlp_ratio = 4.0
        self.drop_p = random.random()
        self.drop_path_p = random.random()

        self.layer = sm.SwinTransformer(
            self.resolution,
            self.num_heads,
            self.window_size,
            self.shift_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
        )

        self.batch_size = 2 ** random.randint(0, 3)
        self.patch_size = self.resolution**2
        self.embed_dim = self.num_heads * random.randint(
            10, 20
        )  # Any multiple of num_heads would be fine

        self.input = tf.random.uniform(
            [self.batch_size, self.patch_size, self.embed_dim], dtype=tf.float32
        )

    def test_window_partition_wrong_inputs(self) -> None:
        wrong_inputs = []

        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution, self.resolution + 1, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Not squared patches
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.patch_size, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Wrong rank

        for input_data in wrong_inputs:
            with self.subTest():
                self.assertRaises(
                    Exception,
                    sm.SwinTransformer.window_partition,
                    input_data,
                    self.window_size,
                )

    def test_window_partition_output(self) -> None:
        input_data = tf.random.uniform(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            dtype=tf.float32,
        )
        output = sm.SwinTransformer.window_partition(input_data, self.window_size)

        self.assertEqual(output.dtype, input_data.dtype)
        self.assertEqual(output.shape[0] % self.batch_size, 0)
        self.assertEqual(output.shape[1], self.window_size)
        self.assertEqual(output.shape[1], output.shape[2])
        self.assertEqual(output.shape[3], self.embed_dim)

    def test_window_reverse_wrong_inputs(self) -> None:
        wrong_inputs = []

        wrong_inputs.append(
            tf.random.uniform(
                [
                    self.batch_size * int((self.resolution / self.window_size)) ** 2,
                    self.window_size,
                    self.window_size + 1,
                    self.embed_dim,
                ],
                dtype=tf.float32,
            )
        )  # Not squared windows
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.window_size**2, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Wrong rank

        for input_data in wrong_inputs:
            with self.subTest():
                self.assertRaises(
                    Exception,
                    sm.SwinTransformer.window_partition,
                    input_data,
                    self.window_size,
                )

    def test_window_reverse_output(self) -> None:
        input_data = tf.random.uniform(
            [
                self.batch_size * int((self.resolution / self.window_size)) ** 2,
                self.window_size,
                self.window_size,
                self.embed_dim,
            ],
            dtype=tf.float32,
        )
        output = sm.SwinTransformer.window_reverse(input_data, self.resolution)

        self.assertEqual(input_data.dtype, output.dtype)
        self.assertEqual(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            output.shape,
        )

    def test_masked_fill_mask_wrong_shape(self) -> None:
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        z = random.randint(1, 10)
        value = 5

        input = tf.random.uniform([x, y], dtype=tf.float32)
        mask = tf.ones([x, y, z], dtype=tf.bool)

        self.assertRaises(
            Exception, sm.SwinTransformer.masked_fill, input, mask, tf.constant(value)
        )

    def test_masked_fill_output(self) -> None:
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        z = random.randint(1, 10)
        value = 5

        input = tf.random.uniform([x, y, z], dtype=tf.float32)
        mask = tf.ones(input.shape, dtype=tf.float32)
        output = sm.SwinTransformer.masked_fill(input, mask, tf.constant(value))

        self.assertEqual(input.shape, output.shape)
        self.assertEqual(input.dtype, output.dtype)
        self.assertEqual(tf.reduce_all(output == value), True)

    def test_build_attn_mask_output(self) -> None:
        output = sm.SwinTransformer.build_attn_mask(
            self.resolution, self.window_size, self.shift_size
        )

        self.assertEqual(tf.float32, output.dtype)
        self.assertEqual((self.resolution / self.window_size) ** 2, output.shape[0])
        self.assertEqual(self.window_size**2, output.shape[1])
        self.assertEqual(output.shape[1], output.shape[2])

    def test_shift_size_bigger_than_window_size(self) -> None:
        self.assertRaises(
            Exception,
            sm.SwinTransformer,
            self.resolution,
            self.num_heads,
            self.window_size,
            self.window_size + 1,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
        )

    def test_resolution_not_evenly_divisible_by_window_size(self) -> None:
        # Find first value that does not evenly divide self.resolution
        wrong_window_size = self.window_size
        while True:
            wrong_window_size += 1

            if self.resolution % wrong_window_size != 0:
                break

        self.assertRaises(
            Exception,
            sm.SwinTransformer,
            self.resolution,
            self.num_heads,
            wrong_window_size,
            int(self.window_size / 2),
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
        )

    def test_output(self) -> None:
        output = self.layer(self.input)

        self.assertEqual(self.input.shape, output.shape)

    def test_wrong_input(self) -> None:
        wrong_inputs = []

        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution**2 - 1, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Wrong num_patches
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.patch_size, self.embed_dim],
                maxval=255,
                dtype=tf.int32,
            )
        )  # Wrong dtype
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution, self.resolution, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Wrong rank

        for input_data in wrong_inputs:
            with self.subTest():
                self.assertRaises(Exception, self.layer, input_data)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer(self.input)

        t_vars = self.layer.trainable_variables

        self.assertEqual(len(t_vars), 13)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer(self.input)

        with tf.GradientTape() as gt:
            output = self.layer(self.input)

        gradients = gt.gradient(output, self.layer.trainable_variables)

        self.assertNotIn(None, gradients)


class TestSwinStage(unittest.TestCase):
    def setUp(self) -> None:
        self.resolution = 2 ** random.randint(3, 6)  # 8-64 px
        self.depth = random.randint(1, 4)
        self.num_heads = random.randint(1, 4)
        self.window_size = int(
            self.resolution / (2 ** random.randint(1, 3))
        )  # Must be evenly divisible by resolution and > 0
        self.mlp_ratio = 4.0
        self.drop_p = random.random()
        self.drop_path_p = random.random()

        self.batch_size = 2 ** random.randint(0, 4)
        self.num_patches = self.resolution**2
        self.embed_dim = self.num_heads * random.randint(
            10, 100
        )  # Must be evenly divisible by num_heads

        self.input = tf.random.uniform(
            [self.batch_size, self.num_patches, self.embed_dim], dtype=tf.float32
        )

    def test_wrong_inputs(self) -> None:
        layer = sm.SwinStage(
            self.resolution,
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=True,
        )

        wrong_input = tf.random.uniform(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            dtype=tf.float32,
        )

        self.assertRaises(Exception, layer, wrong_input)

    def test_output_no_downsample(self) -> None:
        layer = sm.SwinStage(
            self.resolution,
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=False,
        )

        output = layer(self.input)

        self.assertEqual(output.dtype, self.input.dtype)
        self.assertEqual(output.shape, self.input.shape)

    def test_output_downsample(self) -> None:
        layer = sm.SwinStage(
            self.resolution,
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=True,
        )

        output = layer(self.input)

        self.assertEqual(output.dtype, self.input.dtype)
        self.assertEqual(output.shape[0], self.input.shape[0])
        self.assertEqual(output.shape[1], self.input.shape[1] / 4)
        self.assertEqual(output.shape[2], self.input.shape[2] * 2)

    def test_trainable_variables(self) -> None:
        layer = sm.SwinStage(
            self.resolution,
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=True,
        )
        # Build the layer
        layer(self.input)

        t_vars = layer.trainable_variables

        self.assertEqual(len(t_vars), 13 * self.depth + 3)

    def test_gradient(self) -> None:
        layer = sm.SwinStage(
            self.resolution,
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=True,
        )
        # Build the layer
        layer(self.input)

        with tf.GradientTape() as gt:
            output = layer(self.input)

        gradients = gt.gradient(output, layer.trainable_variables)

        self.assertNotIn(None, gradients)


if __name__ == "__main__":
    unittest.main()
