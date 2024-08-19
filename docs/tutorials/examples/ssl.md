# Self Supervised Learning example

In this tutorial we will explain how to train a self-supervised learning model using `Quadra`. Particularly we will focus on the [`Bootstrap your own latent`](https://arxiv.org/abs/2006.07733)(BYOL) algorithm.

## Training
### Dataset 

For self-supervised learning tasks, we will use the same classification dataset structure defined for the `ClassificationDataModule`. In fact the `SSLDataModule` is a subclass of `ClassificationDataModule` and it shares the same API and implementation so it's fairly easy to move from a classification task to a self-supervised learning task.

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
├── test.txt # optional
├── train.txt # optional
└── val.txt # optional
```

The `train.txt`, `val.txt` and `test.txt` files are optional and can be used to specify the list of images to use for training, validation and test. If not specified, the datamodule will base the split using a different parameter. The files should contain the relative path to the image from the dataset root folder. For example, if the dataset is organized as above, the `train.txt` file could be:

```txt
class_0/abc.xyz
...
class_1/abc.xyz
...
class_N/abc.xyz
...
```

Validation is not required but it may be useful to evaluate the embeddings learned by the model during training for example using a linear classifier. The test set will be used to evaluate the model performance at the end of the training.

The default datamodule configuration is found under `configs/datamodule/base/ssl.yaml` and it's defined as follows:

```yaml
_target_: quadra.datamodules.SSLDataModule
data_path: ???
exclude_filter:
include_filter:
seed: ${core.seed}
num_workers: 8
batch_size: 16
augmentation_dataset: null
train_transform: null
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
train_split_file:
val_split_file:
test_split_file:
val_size: 0.3
test_size: 0.1
split_validation: true
class_to_idx:
```

We will make some changes to the datamodule in the experiment configuration file.

### Experiment

First, let's check how base experiment configuration file is defined for BYOL algorithm located in `configs/experiment/base/ssl/byol.yaml`.

```yaml
# @package _global_

defaults:
  - override /datamodule: base/ssl
  - override /backbone: resnet18
  - override /model: byol
  - override /optimizer: lars
  - override /scheduler: warmup
  - override /transforms: byol
  - override /loss: byol
  - override /task: ssl
  - override /trainer: lightning_gpu_fp16
core:
  tag: "run"
  name: "byol_ssl"
task:
  _target_: quadra.tasks.ssl.BYOL

callbacks:
  model_checkpoint:
    _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: "val_acc"
    mode: "max"

trainer:
  devices: [0]
  max_epochs: 500
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 10

datamodule:
  num_workers: 12
  batch_size: 256
  augmentation_dataset:
    _target_: quadra.datasets.TwoAugmentationDataset
    transform:
      - ${transforms.augmentation1}
      - ${transforms.augmentation2}
    dataset: null

scheduler:
  init_lr:
    - 0.4
```

The default configuration can be used to train a `resnet18` model using `lars` as optimizer and a `cosine annealing` scheduler with warmup.
Every 10 epoch we will perform a step of validation using a `KNN` classifier with 20 neighbors (check the model definition for more details). The model will be saved every time the validation accuracy improves and at the end of training.

We will make use of automatic mixed precision to speed up the training process.

Since we are going to use custom dataset for this task we can add a custom experiment configurations under `configs/experiment/custom_experiment/byol.yaml` file.

```yaml
# @package _global_
defaults:
  - base/ssl/byol
  - override /backbone: vit16_tiny # let's use different backbone instead of the resnet
  - _self_

trainer:
  devices: [2] # we may need to use different gpu(s)
  max_epochs: 1000 # let's assume we would like to train for 1000 epochs

datamodule:
  data_path: /path/to/the/dataset
```

!!! note

    If you possess a GPU with bf16 support you can use the `lightning_gpu_bf16` trainer configuration instead of `lightning_gpu_fp16` by overriding the `trainer` section in the experiment configuration file.

    ```yaml
    # @package _global_
    defaults:
      - base/ssl/byol
      - override /backbone: vit16_tiny # let's use different backbone instead of the resnet
      - override /trainer: lightning_gpu_bf16
      - _self_
    ... # rest of the configuration
    ```

### Run

Now we are ready to run our experiment with following command:

```bash
quadra experiment=custom_experiment/byol
```

The output folder should contain the following entries:
```bash
checkpoints  config_resolved.yaml  config_tree.txt  data  deployment_model  main.log
```

The `checkpoints` folder contains the saved `pytorch` lightning checkpoints. The `data` folder contains a joblib version of the datamodule containing all parameters and dataset spits. The `deployment_model` folder contains the model ready for production in the format specified in the `export.types` parameter (default `torchscript`). 

### Run (Advanced) - Changing transformations

In previous example, we have used default transformations defined in the original paper. However, these settings may not be suitable for our dataset. For example, `Gaussian Blur` may destroy important details. In this case, we can extend the experiment configuration file and add our custom transformations.

```yaml
# @package _global_
defaults:
  - base/ssl/byol
  - override /backbone: vit16_tiny # let's use different backbone instead of the resnet
  - _self_

trainer:
  devices: [2] # we may need to use different gpu(s)
  max_epochs: 1000 # let's assume we would like to train for 1000 epochs

datamodule:
  data_path: /path/to/the/dataset

# check configs/transforms/byol.yaml for more details
transforms:
  augmentation1:
    _target_: albumentations.Compose
    transforms:
      - _target_: albumentations.RandomResizedCrop
        height: ${transforms.input_height}
        width: ${transforms.input_width}
        scale: [0.08, 1.0]
      - ${transforms.flip_and_jitter}
      # remove gaussian blur
      # - _target_: albumentations.GaussianBlur
      #   blur_limit: 23
      #   sigma_limit: [0.1, 2]
      #   p: 1.0
      - ${transforms.normalize}
  augmentation2:
    _target_: albumentations.Compose
    transforms:
      - _target_: albumentations.RandomResizedCrop
        height: ${transforms.input_height}
        width: ${transforms.input_width}
        scale: [0.08, 1.0]
      - ${transforms.flip_and_jitter}
      # remove gaussian blur
      # - _target_: albumentations.GaussianBlur
      #   blur_limit: 23
      #   sigma_limit: [0.1, 2]
      #   p: 0.1
      - _target_: albumentations.Solarize
        p: 0.2
      - ${transforms.normalize}
```

During training two different augmentations of the same image will be sampled based on the given parameter and the algorithm will try to match the representations of the two augmentations so picking the right set of transformations is important.