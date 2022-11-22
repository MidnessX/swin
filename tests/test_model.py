import pathlib
import random
import tempfile
import tensorflow as tf
import swin.model as sm
import unittest


class TestSwin(unittest.TestCase):
    def setUp(self) -> None:
        self.ds_batches = 2
        self.batch_size = 2 ** random.randint(0, 2)
        self.img_channels = 3

        self.num_classes = random.randint(1, 10)

        self.window_size = 7
        self.resolution_last_stage = self.window_size * random.randint(1, 3)  # 7-21
        self.num_stages = random.randint(1, 4)
        self.resolutions = [
            self.resolution_last_stage * 2**i for i in range(self.num_stages, 0, -1)
        ]
        self.patch_size = random.randint(2, 5)
        self.img_size = self.resolutions[0] * self.patch_size
        self.depths = [
            random.randint(1, 4) for _ in range(self.num_stages)
        ]  # 1-4 SW-MSA per stage to keep it small enough
        self.num_heads = []
        for i in range(self.num_stages):
            if i == 0:
                self.num_heads.append(random.randint(2, 4))
            else:
                self.num_heads.append(self.num_heads[i - 1] * 2)
        self.embedding_to_head_ratio = 32
        self.embed_dim = self.num_heads[0] * self.embedding_to_head_ratio
        self.drop_rate = random.random()

        self.input = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, self.img_channels],
            dtype=tf.float32,
        )

        self.model = sm.Swin(
            self.input,
            self.num_classes,
            self.patch_size,
            self.embed_dim,
            self.depths,
            self.num_heads,
            self.drop_rate,
        )

    def _build_dataset(self) -> tf.data.Dataset:
        ds_x = tf.random.uniform(
            [
                self.ds_batches,
                self.batch_size,
                self.img_size,
                self.img_size,
                self.img_channels,
            ]
        )
        ds_y = tf.one_hot(
            tf.random.uniform(
                [self.ds_batches, self.batch_size],
                maxval=self.num_classes,
                dtype=tf.int32,
            ),
            self.num_classes,
        )

        ds_x = tf.data.Dataset.from_tensor_slices(ds_x)
        ds_y = tf.data.Dataset.from_tensor_slices(ds_y)
        ds = tf.data.Dataset.zip((ds_x, ds_y))

        return ds

    def _compile_model(self, model: tf.keras.Model) -> None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = tf.keras.metrics.CategoricalAccuracy()

        model.compile(optimizer, loss, metrics)

    def test_model_output(self) -> None:
        output = self.model(self.input)

        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.num_classes)

    def test_model_compile(self) -> None:
        self.model(self.input)

        self._compile_model(self.model)

        self.assertNotEqual(self.model.optimizer, None)
        self.assertNotEqual(self.model.compiled_loss, None)
        self.assertNotEqual(self.model.compiled_metrics, None)

    def test_model_training(self) -> None:
        self.model(self.input)

        self._compile_model(self.model)

        ds = self._build_dataset()

        history = self.model.fit(ds, epochs=1)

        self.assertEqual(len(history.epoch), 1)

    def test_model_restore(self) -> None:
        output_1 = self.model(self.input)

        self._compile_model(self.model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model = pathlib.Path(tmp_dir) / "model"
            self.model.save(tmp_model.as_posix())

            self.model = tf.keras.models.load_model(tmp_model.as_posix())

        output_2 = self.model(self.input)

        diff = tf.abs(output_1 - output_2)
        diff = diff * 0.01  # We tolerate a 1% difference
        diff = tf.floor(diff)
        diff = tf.cast(diff, tf.bool)
        self.assertEqual(tf.reduce_any(diff), False)

    def test_model_restore_config(self) -> None:
        output_1 = self.model(self.input)

        config = self.model.get_config()

        model_copy = sm.Swin.from_config(config)

        output_2 = model_copy(self.input)

        self.assertEqual(output_1.shape, output_2.shape)


if __name__ == "__main__":
    unittest.main()
