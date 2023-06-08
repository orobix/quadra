# Pytorch classification example

In this page, we will show you how to run a classification experiment exploiting Pytorch and Pytorch Lightning to finetune or train from scratch a model on a custom dataset.

This example will demonstrate how to create a custom experiment starting from default settings.

## Training

### Dataset

Let's start with the dataset that we are going to use. Since we are using the classification datamodule, images must be arranged in a folder structure that reflects the classes' partition. 

Suppose we have a dataset with the following structure:
```tree
dataset/
├── class_1
│   ├── abc.xyz
│   └── ...
├── class_2
│   ├── abc.xyz
│   └── ...
├── class_3 
│   ├── abc.xyz
│   └── ...
├── test.txt # optional
├── train.txt # optional
└── val.txt # optional
```

The standard datamodule configuration for classification is found under `configs/datamodule/base/classification.yaml`.

```yaml
_target_: quadra.datamodules.classification.ClassificationDataModule
data_path: ???
num_classes: ${model.num_classes}
exclude_filter: [".ipynb_checkpoints"]
include_filter:
seed: ${core.seed}
num_workers: 8
batch_size: 16
test_size: 0.2
val_size: 0.2
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
train_split_file:
test_split_file:
val_split_file:
label_map:
class_to_idx:
name:
dataset:
  _target_: hydra.utils.get_method
  path: quadra.datasets.classification.ClassificationDataset
```

We give a small description for each tweakable parameter inside the base datamodule config:

- `data_path`: Path to root_folder. "???" denotes mandatory parameters.
- `num_classes`: Number of classes. It is automatically resolved from the model configuration parameter
- `exclude_filter`: If an image path contains one of the strings in this list, it will be ignored.
- `include_filter`: If an image path does not contain one of the strings in this list, it will be ignored.
- `seed`: Seed for experiment reproducibility (if training is on gpu, complete reproducibility can not be ensured).
- `num_workers`: Number of workers used by the dataloaders (same for train/val/test).
- `bath_size`: Batch size for the dataloaders (same for train/val/test).
- `test_size`: If no test_split_file is provided, test_size * len(training_set) will be put in test set.
- `val_size`: If no val_split_file is provided, val_size * len(remaining_training_set) will be put in validation set.
- `label_map`: You can map classes to other ones in order to group more sub-classes into one macro-class.
- `class_to_idx`: Map classes to indexes, if you need to ensure that one specific mapping is respected (otherwise a ordered class_to_idx is built)


### Experiment

Suppose that we want to run the experiment on the given dataset, we can define a config starting from the base config (found under `configs/experiment/base/classification/classification.yaml`).

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
  gradcam: true
  run_test: True
  export_type: [torchscript, pytorch]
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
  freeze_parameters_name:

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

The base experiment will train a resnet18 (by default pretrained on Imagenet) for 200 epochs using Adam as optimizer and reducing the learning rate on plateaus.

We can define a custom experiment starting from the base one, and override the parameters that we want to change. Suppose we create a yaml configuration under `configs/experiment/custom_experiment/torch_classification.yaml` with the following content:

```yaml
# @package _global_
defaults:
  - base/classification/classification
  - override /backbone: vit16_tiny
  - _self_

datamodule:
  num_workers: 12
  batch_size: 32
  data_path: path/to/experiment/dataset
  class_to_idx:
    class_1: 0
    class_2: 1
    class_3: 2

task:
  gradcam: True # Enable gradcam computation during evaluation
  run_test: True # Perform test evaluation at the end of training
  export_type: [torchscript, pytorch]
  report: True 
  output:
    example: True # Generate an example of concordants and discordants predictions for each class
   
model:
  num_classes: 3 # This is very important
  module:
    lr_scheduler_interval: "epoch"
    
backbone:
  model:
    pretrained: True
    freeze: False
  freeze_parameters_name: # Here we could specify a list of layer names to freeze

core:
  tag: "run"
  name: "train_core_name"
  
logger:
  mlflow:
    experiment_name: name_of_the_experiment
    run_name: ${core.name}
```

!!! warning

    Remember to set the mandatory parameters "num_classes" and "data_path".

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/torch_classification
```

This should produce the following output files:

```bash
checkpoints           config_tree.txt  deployment_model  test
config_resolved.yaml  data             main.log
```

Where `checkpoints` contains the pytorch lightning checkpoints of the model, `data` contains the joblib dump of the datamodule with its parameters and dataset split, `deployment_model` contains the model in exported format (default is torchscript), `test` contains the test artifacts.

## Evaluation

### Experiment

The same datamodule specified before can be used for inference. There are different modalities to define the test-set. The simplest one is setting test_size=1.0 (remember the .0) and data_path=path/to/another_root_folder, where "another_root_folder" has the same structure as the root_folder described at the start of this document, but it contains only images you want to use for tests.
Another possibility is to pass a test_split_file to the datamodule config:

```yaml
test_split_file: path/to/test_split_file.txt
```

Where test_split_file is a simple .txt file structured in this way:
```
class_1/image1.png
class_1/image2.png
...
class_2/image1.png
class_2/image2.png
...
```

Where each line contains the relative path to the image from the data_path folder.

The default experiment configuration can be found at `configs/experiment/base/classification/classification_evaluation.yaml`:

```yaml
# @package _global_
defaults:
  - override /datamodule: base/classification
  - override /transforms: default_resize

datamodule:
  num_workers: 6
  batch_size: 32

core:
  tag: "run"
  upload_artifacts: true
  name: classification_evalutation_base

logger:
  mlflow:
    experiment_name: name_of_the_experiment
    run_name: ${core.name}

task:
  _target_: quadra.tasks.ClassificationEvaluation
  gradcam: true
  output:
    example: true
  model_path: ???
```

Given that we don't have to set all the training-related parameters, the evaluation experiment .yaml file will be much simpler, suppose it is saved under `configs/experiment/custom_experiment/torch_classification_evaluation.yaml`:

```yaml
# @package _global_
defaults:
  - base/classification/classification_evaluation
  - _self_

datamodule:
  num_workers: 6
  batch_size: 32
  data_path: path/to/test/dataset
  test_size: 1.0
  class_to_idx:
    class_1: 0
    class_2: 1
    class_3: 2

core:
  tag: "run"
  upload_artifacts: true
  name: eval_core_name

task:
  output:
      example: true
  model_path: path/to/model.pth
```

Notice that we must provide the path to a deployment model file that will be used to perform inferences. In this case class_to_idx is mandatory (we can not infer it from a test-set). We suggest to be careful to set the same class_to_idx that has been used to train the model.

### Run

Just as before, assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/torch_classification_evaluation
```

This will compute the metrics on the test-set and since `example` is set to `true` it will generate an example of concordants and discordants predictions for each class. 