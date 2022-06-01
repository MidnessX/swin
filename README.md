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

# Load the dataset
train_x = ...
train_y =  ...
num_classes = ...

model = SwinT(input[0], num_classes)

# Build the model
model(input[0])

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

9. ``LayerNormalization`` is applied after building patch embeddings[^1].

[^1]: To stay consistent with the content of the paper.

[^2]: In the original implementation this happens when using default arguments.

[^3]: Researchers note in the paper that adding absolute position information to embedding decreases network capabilities.