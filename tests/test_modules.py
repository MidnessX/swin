"""Module containing tests for the modules of the Swin Transformer network."""

import random
import unittest

import tensorflow as tf

import swin.modules as sm


class TestSwinLinear(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = random.randint(1, 5)
        self.img_size = 2 ** random.randint(3, 8)  # 8-256

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
        self.wrong_input_channels = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, 1], dtype=tf.float32
        )

        self.wrong_inputs = [
            self.wrong_input_shape,
            self.wrong_input_channels,
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
        self.assertEqual(output.dtype, self.input.dtype)

    def test_wrong_input(self) -> None:
        for input_data in self.wrong_inputs:
            with self.subTest(f"{input_data.shape}, {input_data.dtype}"):
                self.assertRaises(AssertionError, self.layer, input_data)

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
        self.patches = 2 * random.randint(7, 28)  # Any even number would be fine
        self.embed_dim = 2 ** random.randint(5, 8)  # 32-256

        self.input = tf.random.uniform(
            [self.batch_size, self.patches, self.patches, self.embed_dim],
            dtype=tf.float32,
        )

        self.layer = sm.SwinPatchMerging()

    def test_build_odd_patch_size(self) -> None:
        self.assertRaises(Exception, sm.SwinPatchMerging, 7)  # Any odd number is fine

    def test_output(self) -> None:
        output = self.layer(self.input)

        shape = output.shape

        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[0], self.batch_size)
        self.assertEqual(shape[1], shape[2])
        self.assertEqual(shape[1], self.patches / 2)
        self.assertEqual(shape[3], self.embed_dim * 2)
        self.assertEqual(output.dtype, self.input.dtype)

    def test_wrong_input(self) -> None:
        self.wrong_input_shape = tf.random.uniform(
            [self.batch_size, self.patches**2, self.embed_dim],
            dtype=tf.float32,
        )

        self.wrong_inputs = [
            self.wrong_input_shape,
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

    def test_wrong_probability(self) -> None:
        self.assertRaises(AssertionError, sm.SwinDropPath, -1.2)
        self.assertRaises(AssertionError, sm.SwinDropPath, 2.0)

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
        self.resolution = random.randint(1, 10)
        self.embed_dim = random.randint(1, 100)
        self.input = tf.random.uniform(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            dtype=tf.float32,
        )

    def test_output(self) -> None:
        output = self.layer(self.input)
        shape = output.shape

        self.assertEqual(
            shape[:-1], [self.batch_size, self.resolution, self.resolution]
        )
        self.assertEqual(shape[-1], self.out_features)
        self.assertEqual(output.dtype, self.input.dtype)

    def test_wrong_input(self) -> None:
        wrong_input = tf.random.uniform(
            [self.batch_size, self.resolution**2, self.embed_dim]
        )

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
        )  # Could also be odd, but it simplifies some future operations
        self.num_heads = random.randint(1, 5)
        self.proj_drop_r = random.random()

        self.layer = sm.SwinWindowAttention(self.num_heads, self.proj_drop_r)

        self.batch_size = random.randint(1, 5)
        self.embed_dim = self.num_heads * random.randint(
            5, 10
        )  # Any value divisible by num_heads would be fine
        self.num_patches = self.window_size * random.randint(
            2, 10
        )  # Must be divisible by window_size
        self.num_windows = (self.num_patches // self.window_size) ** 2

        self.input = tf.random.uniform(
            [
                self.batch_size,
                self.num_windows,
                self.window_size,
                self.window_size,
                self.embed_dim,
            ],
            dtype=tf.float32,
        )

        self.attention_mask = sm.SwinTransformer.build_attn_mask(
            self.num_patches, self.window_size, self.window_size // 2
        )

    def test_build_relative_position_index_output(self) -> None:
        output = self.layer.build_relative_position_index(self.window_size)

        self.assertEqual(output.dtype, tf.int32)
        self.assertEqual(output.shape, [self.window_size**2, self.window_size**2])
        self.assertEqual(
            output[0, self.window_size**2 - 1], 0
        )  # Top-right corner must be 0

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
                    self.window_size**2,
                    self.embed_dim,
                ],
                dtype=tf.float32,
            )
        )  # Wrong shape

        if self.num_heads > 1:
            # Incompatible emebedding dimensions. Only add this test when
            # num_heads is greater than 1 or it will fail as n % 1 = 0 for any
            # n.
            wrong_inputs.append(
                tf.random.uniform(
                    [
                        self.batch_size * self.num_windows,
                        self.window_size**2,
                        self.num_heads - 1,
                    ],
                    dtype=tf.float32,
                )
            )

        for input_data in wrong_inputs:
            with self.subTest(f"{input_data.shape}, {input_data.dtype}"):
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
        self.resolution = 2 ** random.randint(3, 6)  # 8-64 patches
        self.num_heads = random.randint(1, 4)
        # Window size must be evenly divisible by resolution and > 0. We choose
        # > 1 to simplify tests where having window_size = resolution would
        # require a lot more code
        self.window_size = self.resolution // (
            2 ** random.randint(1, 2)
        )  # 4-16 patches
        self.shift_size = self.window_size // 2
        self.mlp_ratio = 4.0
        self.drop_p = random.random()
        self.drop_path_p = random.random()

        self.layer = sm.SwinTransformer(
            self.num_heads,
            self.window_size,
            self.shift_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
        )

        self.batch_size = 2 ** random.randint(0, 3)  # 1-8
        self.embed_dim = self.num_heads * random.randint(
            10, 20
        )  # Any multiple of num_heads would be fine

        self.input = tf.random.uniform(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            dtype=tf.float32,
        )

    def test_window_partition_wrong_inputs(self) -> None:
        wrong_inputs = []

        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution, self.resolution + 1, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Non-square patches
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution**2, self.embed_dim],
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
        self.assertEqual(tf.rank(output), 5)
        self.assertEqual(output.shape[0] % self.batch_size, 0)
        self.assertEqual(output.shape[2], output.shape[3])
        self.assertEqual(output.shape[2], self.window_size)
        self.assertEqual(output.shape[4], self.embed_dim)

    def test_window_partition_order(self) -> None:
        batch_size = 1
        resolution = 4
        embed_dim = 1
        window_size = 2

        window_res = resolution // window_size

        input_data = tf.reshape(
            tf.range(batch_size * resolution**2 * embed_dim),
            [batch_size, resolution, resolution, embed_dim],
        )
        output = sm.SwinTransformer.window_partition(input_data, window_size)

        for batch in range(batch_size):
            for i in range(window_res):
                for j in range(window_res):
                    win_idx = i * window_res + j
                    with self.subTest(f"window {win_idx}"):
                        out_win = output[batch, win_idx]
                        true_win = input_data[
                            batch,
                            i * window_size : (i + 1) * window_size,
                            j * window_size : (j + 1) * window_size,
                        ]

                        self.assertTrue(tf.reduce_all(tf.equal(out_win, true_win)))

    def test_window_reverse_wrong_inputs(self) -> None:
        wrong_inputs = []

        wrong_inputs.append(
            tf.random.uniform(
                [
                    self.batch_size,
                    (self.resolution // self.window_size) ** 2,
                    self.window_size,
                    self.window_size + 1,
                    self.embed_dim,
                ],
                dtype=tf.float32,
            )
        )  # Non-square windows
        wrong_inputs.append(
            tf.random.uniform(
                [
                    self.batch_size,
                    (self.resolution // self.window_size) ** 2,
                    self.window_size**2,
                    self.embed_dim,
                ],
                dtype=tf.float32,
            )
        )  # Wrong rank

        for input_data in wrong_inputs:
            with self.subTest(f"{input_data.shape}, {input_data.dtype}"):
                self.assertRaises(
                    Exception,
                    sm.SwinTransformer.window_reverse,
                    input_data,
                    self.window_size,
                )

    def test_window_reverse_output(self) -> None:
        input_data = tf.random.uniform(
            [
                self.batch_size,
                (self.resolution // self.window_size) ** 2,
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

    def test_build_attn_mask_output(self) -> None:
        output = sm.SwinTransformer.build_attn_mask(
            self.resolution, self.window_size, self.shift_size
        )

        self.assertEqual(tf.float32, output.dtype)
        self.assertEqual((self.resolution / self.window_size) ** 2, output.shape[0])
        self.assertEqual(self.window_size**2, output.shape[1])
        self.assertEqual(output.shape[1], output.shape[2])
        self.assertTrue(
            tf.reduce_all(tf.logical_or(output == -100.0, output == 0))
        )  # No value other than 0 or -100 should be present

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
                [self.batch_size, self.resolution, self.resolution - 1, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Non-square patches
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution, self.resolution, self.embed_dim],
                maxval=255,
                dtype=tf.int32,
            )
        )  # Wrong dtype
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution**2, self.embed_dim],
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
        self.resolution = 2 ** random.randint(3, 6)  # 8-64 patches
        self.depth = random.randint(1, 4)
        self.num_heads = random.randint(1, 4)
        self.window_size = self.resolution // (
            2 ** random.randint(1, 3)
        )  # Must be evenly divisible by resolution and > 0
        self.mlp_ratio = 4.0
        self.drop_p = random.random()
        self.drop_path_p = random.random()

        self.batch_size = 2 ** random.randint(0, 4)
        self.embed_dim = self.num_heads * random.randint(
            10, 100
        )  # Must be evenly divisible by num_heads

        self.input = tf.random.uniform(
            [self.batch_size, self.resolution, self.resolution, self.embed_dim],
            dtype=tf.float32,
        )

        self.layer_ds = sm.SwinStage(
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=True,
        )
        self.layer_no_ds = sm.SwinStage(
            self.depth,
            self.num_heads,
            self.window_size,
            self.mlp_ratio,
            self.drop_p,
            self.drop_path_p,
            downsample=False,
        )

    def test_wrong_inputs(self) -> None:
        wrong_inputs = list()

        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution**2, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Wrong rank
        wrong_inputs.append(
            tf.random.uniform(
                [self.batch_size, self.resolution, self.resolution + 1, self.embed_dim],
                dtype=tf.float32,
            )
        )  # Non-square input

        for input_data in wrong_inputs:
            with self.subTest(f"{input_data.shape}, {input_data.dtype}"):
                self.assertRaises(AssertionError, self.layer_ds, input_data)

    def test_output_no_downsample(self) -> None:
        output = self.layer_no_ds(self.input)

        self.assertEqual(output.dtype, self.input.dtype)
        self.assertEqual(output.shape, self.input.shape)

    def test_output_downsample(self) -> None:
        output = self.layer_ds(self.input)

        self.assertEqual(output.dtype, self.input.dtype)
        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[0], self.input.shape[0])
        self.assertEqual(output.shape[1], output.shape[2])
        self.assertEqual(output.shape[1], self.input.shape[1] / 2)
        self.assertEqual(output.shape[3], self.input.shape[3] * 2)

    def test_trainable_variables(self) -> None:
        # Build the layer
        self.layer_ds(self.input)

        t_vars = self.layer_ds.trainable_variables

        self.assertEqual(len(t_vars), 13 * self.depth + 3)

    def test_gradient(self) -> None:
        # Build the layer
        self.layer_ds(self.input)

        with tf.GradientTape() as gt:
            output = self.layer_ds(self.input)

        gradients = gt.gradient(output, self.layer_ds.trainable_variables)

        self.assertNotIn(None, gradients)


if __name__ == "__main__":
    unittest.main()
