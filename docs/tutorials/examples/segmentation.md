# Segmentation Example

In this page, we will show you how to run a segmentation experiment (either binary or multiclass) step by step.

## Training

This example will demonstrate how to create a custom experiment starting from default settings.
### Dataset

Let's start with the dataset we are going to use. Since we are using base segmentation datamodule, we must arrange our images and masks in a folder structure that follows the segmentation datamodule guideline defined in the [segmentation datamodule documentation](../../tutorials/datamodules.md#segmentation).
Imagine that we have a dataset with the following structure:

```tree
dataset/
├── images
│   └── 1.png
├── masks
│   └── 1.png
├── test.txt
├── train.txt
└── val.txt
```

The main difference between multi class segmentation and binary segmentation is that masks contain multiple values. 
For example, if we have a project that uses the following classes:

- `0`: background
- `1`: apple
- `2`: orange
- `3`: banana

Suppose you have a 4x4 image, a possible mask for it may be the following:

$$
\begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 0 & 2 \\
0 & 1 & 0 & 2 \\
0 & 0 & 3 & 3 \\
\end{bmatrix}
$$

The base datamodule configuration file `datamodule/base/segmentation.yaml` is defined as follows:

```yaml
_target_: quadra.datamodules.SegmentationMulticlassDataModule
data_path: ???
idx_to_class:
test_size: 0.3
val_size: 0.3
seed: 42
batch_size: 32
num_workers: 6
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
train_split_file:
test_split_file:
val_split_file:
exclude_good: false
num_data_train:
one_hot_encoding:
```

All the parameters with value `???` must be provided by the user, particularly the `data_path` parameter represents the path to the dataset, since we also have split files we need to provide them as well.
We will do it in the experiment configuration file.

### Backbone

In this example, we will compare models from [`segmentation_models.pytorch`](https://github.com/qubvel/segmentation_models.pytorch) library. There is already a function implemented to load a model from this library acting as a bridge between that library and ours. It is defined as follows under `quadra.modules.backbone`:

```python 
def create_smp_backbone(
    arch: str,
    encoder_name: str,
    freeze_encoder: bool = False,
    in_channels: int = 3,
    num_classes: int = 0,
    **kwargs: Any,
):
    """Create Segmentation.models.pytorch model backbone
    Args:
        arch: architecture name
        encoder_name: architecture name
        freeze_encoder: freeze encoder or not
        in_channels: number of input channels
        num_classes: number of classes
        **kwargs: extra arguments for model (for example classification head).
    """
    model = smp.create_model(
        arch=arch, encoder_name=encoder_name, in_channels=in_channels, classes=num_classes, **kwargs
    )
    if freeze_encoder:
        for child in model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
    return model
```

The default configuration can be found under `configs/backbone/smp.yaml`:

```yaml
model:
  _target_: quadra.modules.backbone.create_smp_backbone
  arch: unet
  encoder_name: resnet18
  encoder_weights: imagenet
  freeze_encoder: True
  in_channels: 3
  num_classes: 1
  activation: null
```

This will create a `Unet` model with `resnet18` encoder and pretrained `imagenet` weights. The encoder will be frozen and the number of input channels will be 3. The number of classes will be changed according to the dataset. The activation function is set to be `null` (converted to `None` in Python) which means that the model will output logits.

!!! note
    These settings are provided by the `segmentation_models.pytorch` library. You can check the other settings from their [documentation](https://github.com/qubvel/segmentation_models.pytorch).

!!! question
    Why do we need to create a backbone function?

    The `segmentation_models.pytorch` library provides a function to load the model from the `segmentation_models.pytorch` library. But we are limited by their customization. If we want to add/try new settings such as freezing the encoder, we need to create a new function to load the model. 

### Experiment

For this example, we will extend base experiment configuration file located in `configs/experiment/base/segmentation/smp_multiclass.yaml` as follows:

```yaml
# @package _global_

defaults:
  - override /datamodule: base/segmentation_multiclass
  - override /model: smp_multiclass
  - override /optimizer: adam
  - override /scheduler: rop
  - override /transforms: default_resize
  - override /loss: smp_dice_multiclass
  - override /task: segmentation
  - override /backbone: smp
  - override /trainer: lightning_gpu

core:
  tag: "run"
  name: "quadra_default"
  upload_artifacts: True

trainer:
  devices: [0]
  max_epochs: 100
  num_sanity_val_steps: 0

datamodule:
  num_workers: 8
  batch_size: 32
```

Let's assume that we would like to change some settings in the default configuration file and add the datamodule parameters. We can extend configuration file as follows creating a custom one in `configs/experiment/segmentation/custom_experiment/smp_multiclass.yaml`:

```yaml
# @package _global_
defaults:
  - base/segmentation/smp_multiclass  # use smp file as default
  - _self_ # use this file as final config

export:
  types: [onnx, torchscript]

backbone:
  model:
    num_classes: 4 # The total number of classes (background + foreground)

task:
  run_test: true # run test after training is completed
  report: false # allows to generate reports
  evaluate: # custom evaluation toggles
    analysis: false # Perform in depth analysis
    
datamodule:
  data_path: /path/to/the/dataset # change the path to the dataset
  batch_size: 64 # update batch size from 32 to 64
  train_split_file: ${datamodule.data_path}/train.txt # Use hydra variable interpolation to create path to the train split file
  test_split_file: ${datamodule.data_path}/test.txt 
  val_split_file: ${datamodule.data_path}/val.txt 
  idx_to_class: # Contains the mapping of all classes without background
    1: "apple"
    2: "orange"
    3: "banana"
  
trainer:
  devices: [3] # change gpu from 0 to 3
  
core:
  upload_artifacts: True # upload artifacts after training is completed (if supported by the current logger)
  name: "custom_segmentation_experiment" # change the name of experiment
```

!!! warning

    When defining the `idx_to_class` dictionary, the keys should be the class index and the values should be the class name. The class index starts from 1.


In the final configuration experiment we have specified the path to the dataset, batch size, split files, GPU device, experiment name and toggled some evaluation options, moreover we have specified that we want to export the model to `onnx` and `torchscript` formats.

By default data will be logged to `Mlflow`. If `Mlflow` is not available it's possible to configure a simple csv logger by adding an override to the file above:
  
```yaml
# @package _global_
defaults:
  - base/segmentation/smp_multiclass  # use smp file as default
  - override /logger: csv # override logger to csv
  - _self_ # use this file as final config

... # The rest of the configuration
```

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=segmentation/custom_experiment/smp_multiclass
```

### Run (Advanced)

Lets assume that you would like to run the experiment with different models and with/without freezing the encoder part of the model, thanks to `hydra` multi-run option we can run as follows:

```bash
quadra experiment=segmentation/custom_experiment/smp_multiclass \
backbone.model.arch=unet,unetplusplus \
backbone.model.encoder_name=resnet18,resnet50 \
backbone.model.freeze_encoder=False,True \
--multirun
```

!!! note

    - Each comma seperated value will be treated as a separate run. In this example will have a total of 8 experiments running. After running this command you may grab a ☕ and wait for the results. 
    - `--multirun` option allows hydra to parse the command-line arguments and create multiple runs.


<!-- ## Creating Model Benchmark -->




