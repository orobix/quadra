# Configuration 

This section explains how the configuration files and folders are structured. It will help you understand how to use and add new configuration files, and where to add them.

!!! warning
    Configuration files heavily depend on the `hydra` library. If you are not familiar with `hydra`, you are strongly advised to read their [documentation](https://hydra.cc/docs/intro/) before using this library.

## Parent Folder Structure

Quadra configurations are divided into macro-categories. These categories can be found under the `configs` folder. The structure of the `configs` folder is the following:

```tree
configs/
├── backbone
├── callbacks
├── core
├── datamodule
├── experiment
├── hydra
├── logger
├── loss
├── model
├── optimizer
├── scheduler
├── task
├── trainer
└── transforms
```

## Config Folders

In this section, we will explain the structure of the config folders. Each folder contains a set of config files or subfolders.
### Backbone

Backbones are the `torch.nn.Module` objects that are used in experiments, generally for feature extraction. Lets Have a look at one example:

```yaml
#configs/backbone/dino_vitb8.yaml
model:
  _target_: quadra.models.classification.TorchHubNetworkBuilder
  repo_or_dir: facebookresearch/dino:main
  model_name: dino_vitb8
  pretrained: true
  freeze: false
  hyperspherical: false
metadata:
  input_size: 224
  output_dim: 768
  patch_size: 8
  nb_heads: 12
```

- **model:** This is the object to instantiate. In this case, it will load a `VitB-8` pretrained model from the torch hub which will be wrapped in a `TorchHubNetworkBuilder` adapter class to make it compatible with our framework.
- **metadata:** This is the object where we store all the parameters related to model. It generally contains useful information for the model, such as input size, output dimension, patch size (for transformers), etc.

Under backbones we may have other kind of models which may not necessary be related to Pytorch, but so far all the implemented tasks are based on Pytorch.

### Callbacks

In this folder we will store all the callbacks that are used in experiments. These callbacks are passed to pytorch-lightning as `trainer.callbacks`.

### Core

Core files contain some global settings for experiments. The default config is the following:

```yaml
version:
  _target_: quadra.get_version
seed: 42
tag: null
name: null
cv2_num_threads: 1
command: "python "
experiment_path: null
upload_artifacts: False
log_level: info
```
For example we can set the seed, decide for a run name, or set the log level.

### Datamodule

Datamodule setting files are used to configure the datamodule which manages the datasets and dataloaders used in experiment. For a detailed explanation on how to implement `DataModule` classes, please refer to the [datamodule documentation](../tutorials/datamodules.md).

Here is the structure of the folder:

```tree
datamodule/
├── base
│   └── ...
└── generic
    └── ...
```

Right now we provide two types of configurations:

- **base:** These are the default configurations for all the experiments. Standard experiments are using this configuration to initialize the datamodules.
- **generic:** These configurations are used to define the datamodules for the generic tasks, which are tasks used to provide examples of how to use the framework.

Let's have a look at one example (`datamodule/base/anomaly.yaml`) configuring the datamodule for anomaly detection:

```yaml
# DataModule class to instantiate
_target_: quadra.datamodules.AnomalyDataModule
# When we find ??? it means that the value must be provided by the user
data_path: ???
category:
num_workers: 8
train_batch_size: 32
test_batch_size: 32
# We can use hydra interpolation to define the value of a variable based on another variable
seed: ${core.seed}
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
phase: train
valid_area_mask:
crop_area:
```

In this case we have all the parameters to instantiate an `AnomalyDataModule` class. Most of the time setting the `data_path` is enough to instantiate the datamodule.

!!! question
    **How can I add a new dataset/datamodule?**

    1. Check if your task is suitable for already defined Datamodule classes defined [here][quadra.datamodules].
        - If it is not suitable, you can create a new Datamodule class `<your-new-task>.py` under `quadra.datamodules`.
        - If it is suitable, you don't need to add a new Datamodule class.
    2. Add new configuration file. For example, if the task is `video_recognition`, we can create a configuration `video_recognition.yaml` in the `datamodule/base` folder. 
### Experiment

The experiment files is the entry-point of the configuration. In here, we combine the different building block configurations and then we add eventual final updates or changes to them. The folder structure is as follows:

```tree
experiment/
├── base
│   ├── anomaly
|   |   └── ...
│   ├── classification
|   |   └── ...
│   ├── segmentation
|   |   └── ...
│   └── ssl
|       └── ...
└── generic
    └── ...
```

Again we have a set of base experiments providing standard configuration for all the tasks and generic ones providing examples of how to use the framework. In this case experiments are divided by the task type.
Let's see an example taking the base experiment for pytorch classification (`experiment/base/classification/classification.yaml`):

```yaml
# @package _global_
defaults:
  - override /backbone: resnet18
  - override /datamodule: base/classification
  - override /loss: cross_entropy
  - override /model: classification
  - override /optimizer: adam
  - override /task: classification
  - override /scheduler: rop
  - override /transforms: default_resize

datamodule:
  num_workers: 8
  batch_size: 32
  data_path: ???

print_config: true

model:
  num_classes: ???
  module:
    lr_scheduler_interval: "epoch"

task:
  lr_multiplier: 0.0
  run_test: True
  export_type: [torchscript]
  report: True
  output:
    example: True

core:
  tag: "run"
  upload_artifacts: true
  name: classification_base_${trainer.max_epochs}

logger:
  mlflow:
    experiment_name: classification_base
    run_name: ${core.name}

backbone:
  model:
    pretrained: True
    freeze: False
    drop_rate: 0.1
  freeze_parameters_name:
    - conv1
    - bn1
    - layer1
    - layer2

trainer:
  precision: 32
  max_epochs: 200
  check_val_every_n_epoch: 1
  log_every_n_steps: 1
  devices: [0]

scheduler:
  patience: 20
  factor: 0.9
  verbose: False
  threshold: 0.01

callbacks:
  early_stopping:
    _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val_loss_epoch
    min_delta: 0.01
    mode: min
    patience: 35
    verbose: false
    stopping_threshold: 0
  model_checkpoint:
    monitor: val_loss_epoch
```

In the experiment configuration we aggregate the various building blocks of the framework using the `defaults` key. In this case we are using the `classification` datamodule, a `resnet18` backbone, the `cross_entropy` loss, the `classification` model (Lightning Module), `adam` as optimizer, the `classification` task, the reduce on plateau (`rop`) scheduler and the `default_resize` transform.

 We can also see that we are overriding some of the parameters of the different modules. For example, we are overriding the `lr_scheduler_interval` of the backbone to be `epoch` instead of `step`. We are also overriding the `max_epochs` of the trainer to be 200 instead of the default value.

The experiment is the most important configuration file as it is the one actually telling the framework what to do!

!!! question
    **How can I create a new experiment extending the default experiment configuration?**

    For example, if you want to create a new pytorch classification experiment starting from the configuration above. You can create a new configuration file `my_custom_experiment.yaml` containing the following lines:

    ```yaml
    # @package _global_
    defaults:
      - base/classification/classification # extend from base classsification
      - override /datamodule: my_custom_datamodule # use custom datamodule
      - override /backbone: vit16_tiny # use a different backbone
      - override /trainer: lighting_multigpu # use a different trainer
      - _self_ # apply the rest of the configuration as final change

    # change the default experiment name
    core:
      name: "my_custom_experiment"

    # use different trainer settings
    trainer:
      devices: [0, 1] # use 2 gpus
      max_epochs: 1000
      num_sanity_val_steps: 0
      precision: 16
      check_val_every_n_epoch: 10
      sync_batchnorm: true
    # use other customizations
    ```

### Hydra

These configuration files manage where and how to create folders or subfolders for experiments and other hydra related configurations.

### Logger

Here we define logger classes for saving the experiment data. Most of the configurations are based on `pytorch_lightning.loggers`. Right now we support the following loggers:

- MLFlowLogger: This is the default logger. It will save the experiment data to an [MLFlow](https://www.mlflow.org/) server.
- TensorBoardLogger: This is a logger that will save the experiment data to a [TensorBoard](https://www.tensorflow.org/tensorboard) server.
- CSVLogger: This is a logger that will simply save the experiment data to a CSV file.

#### Mlflow credentials

The default logging backend for most of the experiments is [https://mlflow.org/](`mlflow`). To use `mlflow` you need to create a `.env` file in the main folder of your project containing the following variables:

```bash
MLFLOW_TRACKING_URI=<url>
MLFLOW_S3_ENDPOINT_URL=<url>
AWS_ACCESS_KEY_ID=<str> # Optional for artifact storage
AWS_SECRET_ACCESS_KEY=<str> # Optional for artifact storage
```

Artifact storage for files such as images or models are using AWS backend at the moment. Other types of third-party `AWS S3` storage providers are also supported. This part is left to user for setting up the infrastructure.

### Loss

The loss functions configurations are defined in this folder.

### Model

The model configurations are the settings we use to instantiate `Lightning Modules`. An example of a model configuration is the one describing a classification module (`model/classification.yaml`):

```yaml
model: ${backbone.model}
num_classes: ???
pre_classifier: null
classifier:
  _target_: torch.nn.Linear
  in_features: ${backbone.metadata.output_dim}
  out_features: ${model.num_classes}
module:
  _target_: quadra.modules.classification.ClassificationModule
  lr_scheduler_interval: "epoch"
  criterion: ${loss}
  gradcam: true
```

We have four main sections in this model configuration:

- `model`: This is the backbone model we use to extract features from the input data.
- `pre_classifier`: This is an optional module that we can add before the classifier layer. It is useful for example to add some MLP before the actual classifier.
- `classifier`: This is the classifier layer that we use to predict the label of the input data.
- `module`: This is the actual `Lightning Module` that we use to train the model. It contains the loss function, the optimizer and the scheduler which will be instantiated using the configurations defined in the `optimizer`, `scheduler` and `loss` sections.

### Optimizer 

Each optimizer file defines how we initialize the training optimizers with their parameters.

### Scheduler

Each scheduler file defines how we initialize the learning rate schedulers with their parameters.

### Task

The tasks are the building blocks containing the actual training and evaluation logic. They are discussed in more details in the [tasks](../tutorials/tasks) section.

### Trainer

In this folder we define the configurations of the classes used to train models, right now we support Lightning based trainers and Sklearn based trainers.

### Transforms

These configurations specify how we apply data augmentation to the datasets.

!!! note
    Usually each transformation configuration creates three data processing pipeline:

    - train_transform
    - val_transform
    - test_transform

The most used transform is the `default_resize` (`transforms/default_resize`) which is used to resize the input images to the size expected by the backbone model and normalize using imagenet mean and std. For example, the `resnet18` backbone expects images of size `[224, 224]` so we use the `default_resize` transform to resize the input images to this size.

```yaml
defaults:
  - default
  - _self_

input_height: 224
input_width: 224

standard_transform:
  _target_: albumentations.Compose
  transforms:
    - _target_: albumentations.Resize
      height: ${transforms.input_height}
      width: ${transforms.input_width}
      interpolation: 2
      always_apply: True
    - ${transforms.normalize}

train_transform: ${transforms.standard_transform}
val_transform: ${transforms.standard_transform}
test_transform: ${transforms.standard_transform}

name: default_resize
```

Basically all the transforms have an `input_height` and `input_width` parameter which are used to resize the input images to the size expected by the backbone model. Right now we support only [Albumentations](https://albumentations.ai/) based transforms.

We make large use of hydra variable interpolation to make the configuration files more readable and avoid repeating the same parameters over and over.