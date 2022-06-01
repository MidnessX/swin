# Swin Transformer

## Description

![Swin Transformer architecture](https://github.com/microsoft/Swin-Transformer/blob/3b0685bf2b99b4cf5770e47260c0f0118e6ff1bb/figures/teaser.png)

This is a TensorFlow 2.0 implementation of the [Swin Transformer architecture](https://arxiv.org/abs/2103.14030).

It is built using the Keras API following best practices, such as allowing complete serialization and deserialization of custom layers and deferring weight creation until the first call with real inputs.

This implementation is inspired by the [official version](https://github.com/microsoft/Swin-Transformer) offered by authors of the paper, while simultaneously improving in some areas such as shape and type checks.

## Installation

Clone the repository:
```bash
git clone git@github.com:MidnessX/swin.git
```
Enter into the directory:
```bash
cd swin
```
Install the package via:
```bash
pip install -e .
```

## Usage

Class ``Swin`` in  ``swin.model`` is a subclass of ``tf.keras.Model``, so you can instantiate Swin Transformers and train them through well known interface methods, such as ``compile()``, ``fit()``, ``save()``.

The only remark is the first argument to the ``Swin`` class constructor, which is expected to be a ``tf.Tensor`` object or equivalent, such as a symbolic tensor produced by ``tf.keras.Input``.
This tensor is only used to determine the shape of future inputs and can be an example coming from your dataset or any random tensor sharing its shape.

For convenience, ``swin.model`` also includes classes for variants of the Swin architecture described in the article (``SwinT``, ``SwinS``, ``SwinB``, ``SwinL``) which initialize a ``Swin`` object with the variant's parameters.

## Example

```python
import tensorflow as tf

from swin.model import SwinT

# Load the dataset as a list of mini batches
train_x = ...
train_y =  ...
num_classes = ...

# Take a mini batch from the dataset to build the model
mini_batch = train_x[0]

model = SwinT(mini_batch, num_classes)

# Build the model by calling it for the first time
model(mini_batch)

# Compile the model
model.compile(
    loss=tf.keras.losses.SGD(learning_rate=1e-3, momentum=0.9),
    optimizer=tf.keras.optimizers.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

# Train the model
history = model.fit(train_x, train_y, epochs=300)

# Save the trained model
model.save("path/to/model/directory")
```

## Notes

- The input type accepted by the model is ``tf.float32``. Any pre-processing of data should include a conversion step of images from ``tf.uint8`` to ``tf.float32`` if necessary.

- Swin architectures have many parameters, so training them is not an easy task. Expect a lot of trial & error before honing in correct hyperparameters.

- ``SwinModule`` layers place the dimensionality reduction layer (``SwinPatchMerging``) after transformer layers (``SwinTransformer``), rather than before as found in the paper. This choice is to maintain consistency with the original network implementation.
 
## Testing

Test modules can be found under the ``tests`` folder of this repository.
They can be executed to test the expected functionality of custom layers for the Swin architecture, as well as basic functionalities of the whole model.

Admittedly these tests could be expanded and further improved to cover more cases, but they should be enough to verify general functionality.

## Assumptions and simplifications

While implementing the Swin Transformer architecture a number of assumptions and simplifications have been made:

1. Input images must have 3 channels.

2. The size of windows in (Shifted) Windows Multi-head Attention is fixed to 7[^1].

3. The ratio of hidden to output neurons in ``SwinMlp`` layers is fixed to 4[^1].

4. A learnable bias is added to ``queries``, ``keys`` and ``values`` when computing (Shifted) Window Multi-head Attention[^2].

5. ``queries`` and ``keys`` are scaled by a factor of ``head_dimension**-0.5``[^1].

6. No dropout is applied to attention heads[^2].

7. The probability of the Stochastic Depth computation-skipping technique during training is fixed to 0.1[^2].

8. No absolute position information is included in embeddings[^3].

9. ``LayerNormalization`` is applied after building patch embeddings[^2].

[^1]: To stay consistent with the content of the paper.

[^2]: In the original implementation this happens when using default arguments.

[^3]: Researchers note in the paper that adding absolute position information to embedding decreases network capabilities.

## Choosing parameters

### Dependencies

If using the base class (``Swin``), it is necessary to provide a series of parameters to instantiate the model.
The choice of these values is important and a series of dependencies exist between them.

The size of windows (``window_size``) used during (Shifted) Windows Multi-head Self Attention is the starting point and, as stated in the section about [assumptions](https://github.com/MidnessX/swin#assumptions-and-simplifications), it is fixed to ``7`` (as in the original paper).

The resolution of inputs to network stages, expressed as the number of patches along each axis, must be a multiple of ``window_size`` and gets halved by every stage through ``SwinPatchMerging`` layers.
The suggestion is to choose a resolution for the final stage and multiply it by ``2`` for every stage in the desired model, obtaining the input resolution of the first stage (``resolution_stage_1``).

Input images to the ``Swin`` model must be squares, with their height/width given by multiplying ``resolution_stage_1`` with the desired size of patches (``patch_size``).

The number of ``SwinTransformer`` layers in each stage (``depths``) is arbitrary.

The number of transformer heads (``num_heads``) should instead double at each stage.
Authors of the paper use a fixed ratio between embedding dimensions and the number of heads in each stage of the network, amounting to ``32``.
This means that, chosen the number of transformer heads in the first stage, it should be multiplied by ``32`` to obtain ``embed_dim``.

The following example should help clarify these concepts.

### Parameter choice example

Let's imagine we want a Swin Transformer having ``3`` stages.
The last stage (``stage_3``) should receive inputs of ``14x14`` patches (``14 = window_size * 2``); this also means that ``stage_2`` receives inputs of ``28x28`` patches and ``stage_1`` of ``56x56``.

We want to convert our images into patches having size ``6x6``, so images should have size ``56 * 6 = 336``.

Our network will have ``2`` transformers in the first stage, ``4`` in the second and ``2`` in the third.
We choose ``4`` heads for the first stage and thus the second one will have ``8`` heads while the third ``16``.

With these numbers we can derive the size of embeddings used in the first stage by multiplying ``32`` by ``4``, giving us ``128``.

Summarizing, we have:

- ``image_size = 336``
- ``patch_size = 6``
- ``embed_dim = 128``
- ``depths = [2, 4, 2]``
- ``num_heads = [4, 8, 16]``