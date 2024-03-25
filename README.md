# Swin Transformer

## Description

![Swin Transformer architecture](https://github.com/microsoft/Swin-Transformer/blob/3b0685bf2b99b4cf5770e47260c0f0118e6ff1bb/figures/teaser.png)

This is a Kears/TensorFlow 2.0 implementation of the [Swin Transformer architecture](https://arxiv.org/abs/2103.14030) inspired by the  official Pytorch [code](https://github.com/microsoft/Swin-Transformer).

It is built using the Keras API following best practices, such as allowing complete serialization and deserialization of custom layers and deferring weight creation until the first call with real inputs.

## Installation

Clone the repository:
```bash
git clone git@github.com:MidnessX/swin.git
```
Enter into it:
```bash
cd swin
```
Install the package via:
```bash
pip install .
```

## Usage

Class ``Swin`` in  ``swin.model`` is a subclass of ``tf.keras.Model``, so you can instantiate Swin Transformers and train them through well known interface methods, such as ``compile()``, ``fit()``, ``save()``.

For convenience, ``swin.model`` also includes classes for variants of the Swin architecture described in the article (``SwinT``, ``SwinS``, ``SwinB``, ``SwinL``) which initialize a ``Swin`` object with the variant's parameters.

## Example

```python
import tensorflow.keras as keras
from swin import Swin

# Dataset loading, omitted for brevity
x = [...]
y = [...]
num_classes = [...]

model = Swin(num_classes)

model.compile(
    optimizer=keras.optimizers.AdamW(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.fit(
    x,
    y,
    epochs=1000,
)

model.save("path/to/model/directory")
```

## Notes

This network has been built to be consistent with its [official  Pytorch implementation](https://github.com/microsoft/Swin-Transformer).
This translates into the following statements:

- The ratio of hidden to output neurons in MLP blocks is set to 4.
- Projection of input data to obtain `Q`, `K`, and `V` includes a bias term in all transformer blocks.
- `Q` and `K` are scaled by `sqrt(d)`, where `d` is the size of `Q` and `K`.
- No _Dropout_ is applied to attention heads.
- [_Stochastic Depth_](https://arxiv.org/pdf/1603.09382.pdf) is applied to randomly disable patches after the attention computation, with probability set to 10%.
- No absolute position information is added to embeddings.
- _Layer Normalizaton_ is applied to embeddings.
- The extraction of patches from images and the generation of embeddings both happen in the `SwinPatchEmbeddings` layer.
- Patch merging happens at the end of each stage, rather than at the beginning.
  This simplifies the definition of layers and does not change the overall architecture.

Additionally, the following decisions have been made to simplify development:

- The network only accepts square `tf.float32` images with 3 channels as inputs (i.e. height and width must be identical).
- No padding is applied to embeddings during the SW-MSA calculation, as their size is assumed to be a multiple of window size.

## Choosing parameters

When using any of the subclasses (``SwinT``, ``SwinS``, ``SwinB``, ``SwinL``), the architecture is fixed to their respective variants found in the paper.

When using the `Swin` class directly, however, you can customize the resulting architecture by specifing all the network's parameters.
This sections provides an overview of the dependencies existing between these parameters.

- Each stage has an input with shape `(batch_size, num_patches, num_patches, embed_dim)`.
`num_patches` must be a multiple of `window_size`.
- Each stage halves the `num_patches` dimension by merging four adjacent patches together.
  It can be easier to choose a desired number of patches in the last stage and multiply it by 2 for every stage in the network to obtain the initial `num_patches` value.
- By multiplying `num_patches` by `patch_size` you can find out the size in pixels of input images.
- `embed_dim` must be a multiple of `num_heads` for every stage.
- The number of transformer blocks in each stage can be set freely, as they do not alter the shape of patches.

To better understand how to choose network parameters, consider the following example:

1. The depth is set to 3 stages.
2. Windows are set to be 8 patches wide (i.e. `window_size = 8`).
3. The last stage should have a `2 * window_size = 16` patch-wide input.
   This means that the input to the second stage and the first stage will be 32x32 and 64x64 patch-wide respectively.
4. We require each patch to cover a 6x6 pixel area, so the initial image will be `num_patches * 6 = 64 * 6 = 384` pixel wide.
5. For the first stage, we choose 2 attention layers; 4 for the second, and 2 for the third.
6. The number of attention heads is set to 4.
   This implies that there will be 8 attention heads in the second stage and 16 attention heads in the third stage.
7. Using the value found in the Swin paper, the `embed_dim / num_heads` ratio is set to 32, leading to an initial `embed_dim` of `32 * 4 = 128`.

Summarizing, this is equal to:

- `image_size  = 384`
- `patch_size = 6`
- `window_size = 8`
- `embed_dim = 128`
- `depths = [2, 4, 2]`
- `num_heads = [4, 8, 16]`

## Testing

Test modules can be found under the ``tests`` folder of this repository.
They can be executed to verify the expected functionality of custom layers for the Swin architecture, as well as basic functionalities of the whole model.

You can run them with the following command:
```bash
python -m unittest discover -s ./tests
```

## Extras

To better understand how SW-MSA works, a Jupyter notebook found in the `extras` folder can be used to visualize window partitioning, traslation and mask construction.

