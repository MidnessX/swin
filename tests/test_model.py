import pathlib
import random
import tempfile
import unittest

import tensorflow as tf

import swin.model as sm

MODEL_VARIANTS = [sm.SwinT, sm.SwinS, sm.SwinB, sm.SwinL]
KERAS_MODEL_FORMATS = ["tf", "keras_v3"]
RESTORED_MODEL_TOLERANCE = 1e-2  # A 1% difference between the original model's output and its restored-from-disk counterpart is tolerated


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
        self.drop_path_rate = random.random()

        self.input = tf.random.uniform(
            [self.batch_size, self.img_size, self.img_size, self.img_channels],
            dtype=tf.float32,
        )

        self.model = sm.Swin(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
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

    def _restored_model_output(
        self, model: tf.keras.Model, inputs: tf.Tensor, fmt: str
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = pathlib.Path(tmp_dir)

            if fmt.startswith("keras"):
                model_path = (
                    model_path / "model.keras"
                )  # The keras format requires a path to a .keras file

            model.save(model_path, save_format=fmt)

            model = tf.keras.models.load_model(model_path)

        return model(inputs)

    def test_model_output(self) -> None:
        output = self.model(self.input)

        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.num_classes)

    def test_model_variants_output(self) -> None:
        image_size = 224

        for variant in MODEL_VARIANTS:
            with self.subTest(f"Variant {variant}"):
                model = variant(num_classes=self.num_classes, drop_rate=self.drop_rate)
                inputs = tf.random.uniform([self.batch_size, image_size, image_size, 3])
                output = model(inputs)

                self.assertEqual(output.shape[0], self.batch_size)
                self.assertEqual(output.shape[1], self.num_classes)

    def test_model_variants_restore(self) -> None:
        variant = sm.SwinT  # Use SwinT as it's faster. Other variants are identical.

        model = variant(self.num_classes)
        x = tf.random.uniform([1, 224, 224, 3])

        output_1 = model(x)

        self._compile_model(model)

        for fmt in KERAS_MODEL_FORMATS:
            with self.subTest(f"{variant.__name__} model, {fmt} format"):
                output_2 = self._restored_model_output(model=model, inputs=x, fmt=fmt)
                self.assertEqual(
                    tf.reduce_all(
                        tf.raw_ops.ApproximateEqual(
                            x=output_1, y=output_2, tolerance=RESTORED_MODEL_TOLERANCE
                        )
                    ),
                    True,
                )

    def test_model_custom_window_size_output(self) -> None:
        depths = [2, 4, 2]
        num_heads = [4, 8, 16]
        patch_size = 6
        window_size = 8
        embed_dim = num_heads[0] * 32
        img_size = 384
        num_classes = random.randint(1, 10)
        batch_size = 2 ** random.randint(1, 3)

        inputs = tf.random.uniform([batch_size, img_size, img_size, 3])
        model = sm.Swin(
            num_classes=num_classes,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
        )

        output = model(inputs)

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], num_classes)

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

        for fmt in KERAS_MODEL_FORMATS:
            with self.subTest(f"{fmt} format"):
                output_2 = self._restored_model_output(
                    model=self.model, inputs=self.input, fmt=fmt
                )

                self.assertEqual(
                    tf.reduce_all(
                        tf.raw_ops.ApproximateEqual(
                            x=output_1, y=output_2, tolerance=RESTORED_MODEL_TOLERANCE
                        )
                    ),
                    True,
                )

    def test_model_restore_config(self) -> None:
        output_1 = self.model(self.input)

        config = self.model.get_config()

        model_copy = sm.Swin.from_config(config)

        output_2 = model_copy(self.input)

        self.assertEqual(output_1.shape, output_2.shape)


if __name__ == "__main__":
    unittest.main()
