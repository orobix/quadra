# Pytorch multilabel classification example

In this page, we will show you how to run a multilabel classification experiment exploiting Pytorch and Pytorch Lightning to finetune or train from scratch a model on a custom dataset.

This example will demonstrate how to create a custom experiment starting from default settings.

## Training

### Dataset

Let's start with the dataset that we are going to use. Since we are using the multilabel classification datamodule we must follow a precise structure for the dataset.

```tree
dataset/
├── images
│   ├── abc.xyz
│   └── ...
├── samples.txt # optional
├── test.txt # optional
├── train.txt # optional
└── val.txt # optional
```

Either `samples.txt` or both `train.txt` and `test.txt` must be provided. If `samples.txt` is provided, it will be used to split the dataset into train/val/test based on the datamodule parameters. Otherwise, `train.txt` and `val.txt` will be used to split the dataset into train/val and `test.txt` will be used to create the test set.

The content of the files is the same for both `samples.txt` and `train.txt`/`val.txt`/`test.txt`:

```txt
images/abc.xyz,class_1,class_2
images/abc_2.xyz,class_3
```

So the first column is the path to the image, while the other columns are the labels associated with that image. The labels must be separated by a comma.

The standard datamodule configuration for classification is found under `configs/datamodule/base/multilabel_classification.yaml`.

```yaml
_target_: quadra.datamodules.MultilabelClassificationDataModule
data_path: ???
images_and_labels_file: null
train_split_file: null
test_split_file: null
val_split_file: null
dataset:
  _target_: hydra.utils.get_method
  path: quadra.datasets.classification.MultilabelClassificationDataset
num_classes: null
num_workers: 8
batch_size: 64
test_batch_size: 64
seed: ${core.seed}
val_size: 0.2
test_size: 0.2
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
class_to_idx: null
```

The most important parameters are:
- `data_path`: the path to the dataset folder
- `images_and_labels_file`: the path to the `samples.txt` file
- `train_split_file`: the path to the `train.txt` file
- `test_split_file`: the path to the `test.txt` file
- `val_split_file`: the path to the `val.txt` file

### Experiment

Suppose that we want to run the experiment on the given dataset, we can define a config starting from the base config (found under `configs/experiment/base/classification/multilabel_classification.yaml`).

```yaml
# @package _global_
defaults:
  - override /backbone: resnet18
  - override /datamodule: base/multilabel_classification
  - override /loss: asl
  - override /model: multilabel_classification
  - override /optimizer: adam
  - override /task: classification
  - override /scheduler: rop
  - override /transforms: default_resize

datamodule:
  num_workers: 8
  batch_size: 32
  test_batch_size: 32

print_config: true

core:
  tag: "run"
  test_after_training: true
  upload_artifacts: true
  name: multilabel-classification

logger:
  mlflow:
    experiment_name: ${core.name}
    run_name: ${core.name}

backbone:
  model:
    pretrained: True
    freeze: False

trainer:
  devices: [0]
  max_epochs: 300
  check_val_every_n_epoch: 1
```

The base experiment will train a resnet18 (by default pretrained on Imagenet) for 300 epochs using Adam as optimizer and reducing the learning rate on plateaus. In we give a look inside the model configuration we will find that on top of the backbone there is a simple Linear layer mapping the output of the backbone to the number of classes.

We can define a custom experiment starting from the base one, and override the parameters that we want to change. Suppose we create a yaml configuration under `configs/experiment/custom_experiment/torch_multilabel_classification.yaml` with the following content:

```yaml
# @package _global_
defaults:
  - base/classification/multilabel_classification
  - override /backbone: vit16_tiny
  - _self_

datamodule:
  num_workers: 12
  batch_size: 32
  data_path: path/to/experiment/dataset
  images_and_labels_file: ${datamodule.data_path}/samples.txt # We make use of hydra variable interpolation
  class_to_idx:
    class_1: 0
    class_2: 1
    class_3: 2

model:
  classifier:
    out_features: 3 # This is very important as it defines the number of classes

task:
  run_test: True # Perform test evaluation at the end of training
  report: False 
  output:
    example: False 

  
logger:
  mlflow:
    experiment_name: name_of_the_experiment
    run_name: ${core.name}
```

!!! warning

    At the current time the report generation is not supported for multilabel classification.

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/torch_multilabel_classification
```

This should produce the following output files:

```bash
checkpoints  config_resolved.yaml  config_tree.txt  data  deployment_model  main.log
```

Where `checkpoints` contains the pytorch lightning checkpoints of the model, `data` contains the joblib dump of the datamodule with its parameters and dataset split, `deployment_model` contains the model in exported format (default is torchscript).

