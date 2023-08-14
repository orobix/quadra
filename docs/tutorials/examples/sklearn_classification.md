# Sklearn classification example

In this page, we will show you how to train an Sklearn classifier using a Pytorch feature extractor.

This example will demonstrate how to create custom experiments starting from default settings.

## Training
### Dataset

Let's start with the dataset that we are going to use. Since we are using the base datamodule, we need to organize the data in a specific way. The base datamodule expects the following structure:

```tree
dataset/
├── class_0 
│   ├── abc.xyz
│   └── ...
├── class_1
│   ├── abc.xyz
│   └── ...
├── class_N 
│   ├── abc.xyz
│   └── ...
├── None 
│   ├── abc.xyz
│   └── ...
├── train.txt # optional
└── test.txt # optional
```

!!! note

    Only for inference a None class representing unknown data can be used. If a folder with the same name during training it will be interpreted as a class.

The `train.txt` and `test.txt` files are optional and can be used to specify the list of images to use for training (validation) and testing. If not specified, the datamodule will base the split using a different parameter. The files should contain the relative path to the image from the dataset root folder. For example, if the dataset is organized as above, the `train.txt` file could be:

```txt
class_0/abc.xyz
...
class_1/abc.xyz
...
class_N/abc.xyz
...
```

The standard datamodule configuration for this kind of task is found under `datamodule/base/sklearn_classification.yaml`.

```yaml
_target_: quadra.datamodules.SklearnClassificationDataModule
data_path: ???
exclude_filter:
include_filter:
val_size: 0.3
class_to_idx:
label_map:
seed: ${core.seed}
batch_size: 32
num_workers: 8
train_transform: ${transforms.train_transform}
val_transform: ${transforms.val_transform}
test_transform: ${transforms.test_transform}
roi:
n_splits: 1
phase:
cache: false
limit_training_data:
train_split_file:
test_split_file:
```

The only required parameter is `data_path` which should point to the dataset root folder. The other parameters can be used to customize the datamodule behavior, the most important parameters are:

- `val_size`: The percentage of the dataset to use for validation (if test_split_file is not specified)
- `class_to_idx`: A dictionary mapping class names to class indexes
- `label_map`: A dictionary mapping groups of classes to a single class (E.g. "good": ["class_1", "class_2"]), it may be useful for testing different scenarios or simplify the classification task
- `roi`: Optional region of intereset in the following format (x_upper_left, y_upper_left, x_bottom_right, y_bottom_right)
- `n_splits`: The number of splits to use to partition the dataset, if 1 the dataset will be split in train and test, if > 1 cross-validation will be used
- `cache`: If cross validation is used is it possible to cache the features extracted from the backbone to speed up the process
- `limit_training_data`: If specified, the datamodule will use only a subset of the training data (useful for debugging)
- `train_split_file`: If specified, the datamodule will use the given file to create the train dataset (which will be the base for validation)
- `test_split_file`: If specified, the datamodule will use the given file to create the test dataset

No matter if cross validation or standard train/test split is used, the final model will be trained on the whole training dataset, the splits are only used to validate the model.

### Experiment

Suppose that we want to run the experiment on the given dataset, we can define a config starting from the base config:
```yaml
# @package _global_

defaults:
  - override /model: logistic_regression
  - override /transforms: default_resize
  - override /task: sklearn_classification
  - override /backbone: dino_vitb8
  - override /trainer: sklearn_classification
  - override /datamodule: base/sklearn_classification

export:
  types: [pytorch, torchscript]
  
backbone:
  model:
    pretrained: true
    freeze: true

core:
  tag: "run"
  name: "sklearn-classification"

datamodule:
  num_workers: 8
  batch_size: 32
  phase: train
  n_splits: 1
```

By default the experiment will use dino_vitb8 as backbone, resizing the images to 224x224 and training a logistic regression classifier. Setting the `n_splits` parameter to 1 will use a standard 70/30 train/validation split (given the parameters specified in the base datamodule) instead of cross validation.
It will also export the model in two formats, "torchscript" and "pytorch".

An actual configuration file based on the above could be this one (suppose it's saved under `configs/experiment/custom_experiment/sklearn_classification.yaml`):

```yaml
# @package _global_

defaults:
  - base/classification/sklearn_classification
  - override /backbone: resnet18
  - _self_

core:
  name: experiment-name

export:
  types: [pytorch, torchscript]

datamodule:
  data_path: path_to_dataset
  batch_size: 64
  class_to_idx:
    class_0: 0
    class_1: 1
    class_2: 2
  n_splits: 5
  train_split_file: ${datamodule.data_path}/train.txt
  test_split_file: ${datamodule.data_path}/test.txt


task:
  device: cuda:0
  output:
    folder: classification_experiment
    save_backbone: true
    report: true
    example: true
    test_full_data: true
```

This will train a logistic regression classifier using a resnet18 backbone, resizing the images to 224x224 and using a 5-fold cross validation. The `class_to_idx` parameter is used to map the class names to indexes, the indexes will be used to train the classifier. The `output` parameter is used to specify the output folder and the type of output to save. The `export.types` parameter can be used to export the model in different formats, at the moment `torchscript`, `onnx` and `pytorch` are supported.
Since `save_backbone` is set to true, the backbone (in torchscript and pytorch format) will be saved along with the classifier. `test_full_data` is used to specify if a final test should be performed on all the data (after training on the training and validation datasets).

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/sklearn_classification
```

This will run the experiment training a classifier and saving metrics and reports under the `classification_experiment` folder.

The output folder should contain the following entries:
```bash
classification_experiment    classification_experiment_3  data
classification_experiment_0  classification_experiment_4  deployment_model
classification_experiment_1  config_resolved.yaml         main.log
classification_experiment_2  config_tree.txt              test
```

Each `classification_experiment_X` folder contains the metrics for the corresponding fold while the `classification_experiment` folder contains the metrics computed aggregating the results of all the folds.

The `data` folder contains a joblib version of the datamodule containing parameters and splits for reproducibility. The `deployment_model` folder contains the backbone exported in torchscript format if `save_backbone` to true alongside the joblib version of trained classifier. The `test` folder contains the metrics for the final test on all the data after the model has been trained on both train and validation.

## Evaluation
The same datamodule specified before can be used for inference by setting the `phase` parameter to `test`. 

### Experiment

The default experiment config is found under `configs/experiment/base/classification/sklearn_classification_test.yaml`.

```yaml
# @package _global_

defaults:
  - override /transforms: default_resize
  - override /task: sklearn_classification_test
  - override /trainer: sklearn_classification
  - override /datamodule: base/sklearn_classification

core:
  tag: run
  name: sklearn-classification-test

datamodule:
  num_workers: 8
  batch_size: 32
  phase: test
```

An actual configuration file based on the above could be this one (suppose it's saved under `configs/experiment/custom_experiment/sklearn_classification_test.yaml`):

```yaml
# @package _global_
defaults:
  - base/classification/sklearn_classification_test
  - _self_

core:
  name: experiment-test-name

datamodule:
  data_path: path_to_test_dataset
  batch_size: 64

task:
  device: cuda:0
  gradcam: true
  output:
    folder: classification_test_experiment
    report: true
    example: true
  model_path: ???
```

This will test the model trained in the given experiment on the given dataset. The experiment results will be saved under the `classification_test_experiment` folder. If gradcam is set to True, original and gradcam results will be saved during the generate_report().
Model_path must point to a model file. It could either be a '.pt'/'.pth' or a backbone_config '.yaml' file.

### Run

Same as above, assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/sklearn_classification_test
```