# Anomaly detection example

In this page, we will show you how to run anomaly detections experiment exploiting the functionality of the anomalib
library.

## Training

This example will demonstrate how to create custom experiments starting from default settings.
### Dataset

Let's start with the dataset that we are going to use. Since we are using the base anomaly datamodule, images
and masks must be arranged in a folder structure that follows the anomaly datamodule guidelines defined in the 
[anomaly datamodule documentation](../../tutorials/datamodules.md#anomaly-detection). 
For this example, we will use the `mnist` dataset (using 9 as good and all the other number as anomalies), the dataset will be automatically downloaded by the generic experiment described next.

```tree
MNIST/
├── train 
│   └── good
│       └── xyz.png
└── test
  ├── good
  │   └── xyz.png
  ├── 0
  │   └── xyz.png
  ├── 1
  │   └── xyz.png
  └── ...
      └── xyz.png
```

MNIST doesn't have ground truth masks for defects, by default we will use empty masks for good images and full white masks for anomalies.

The standard datamodule configuration for anomaly is found under `datamodule/base/anomaly.yaml`.

```yaml
_target_: quadra.datamodules.AnomalyDataModule
data_path: ???
category:
num_workers: 8
train_batch_size: 32
test_batch_size: 32
seed: ${core.seed}
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
phase: train
mask_suffix:
valid_area_mask:
crop_area:
```

But for the `mnist` example we will use the generic datamodule configuration under `datamodule/generic/mnist/anomaly/base.yaml`.

```yaml
_target_: quadra.generic.mnist.MNISTAnomalyDataModule
data_path: ${oc.env:HOME}/.quadra/datasets/MNIST
good_number: 9
num_workers: 8
limit_data: 100
train_batch_size: 32
test_batch_size: 32
seed: ${core.seed}
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
phase: train
valid_area_mask:
crop_area:
```

The MNISTDataModule will automatically download the dataset and create the folder structure described above under the `data_path` directory.

### Anomaly detection techniques
At the current stage, six methods taken from the anomalib library are available for anomaly detection (descriptions are taken or readaptaded from the anomalib documentation):

- [PADIM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/padim): Padim extends the concepts of DFM fitting gaussian distributions on a lot of feature vectors extracted from
intermediate layer of the network, thus retaining spatial information and allowing outputting anomaly maps that can be
used to segment the images. From the conducted experiments is a strong model that fits well on multiple datasets.
- [Patchcore](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/patchcore): Similarly to PADIM feature extraction is done on intermediate layers, that are then pooled to 
furtherly extend the receptive field of the model. Features from the training images are used to create a so called
"memory bank" that is used as base dataset to perform KNN in the inference stage. This technique should be even
stronger than PADIM but it is a bit slower, generally I would use it if PADIM fails.
- [CFLOW](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/cflow): While the previous techniques generally performs a single forward pass to extract features and fit some kind
of model, CFLOW is more similar to the standard neural network training procedure, where the model is trained for 
multiple epochs. The model is fairly complex and is based on the concept of "Normalizing flow" which main idea is
to transform the model complex distribution into a simpler one using Invertible Neural Networks. The concept of 
normalizing flow looks very promising and many recent papers are achieving good results with it. But so far conducted
experiments have shown that the previous models are better and faster to train.
- [Fastflow](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/fastflow): FastFlow is a two-dimensional normalizing flow-based probability distribution estimator. It can be used as a plug-in module with any deep feature extractor, such as ResNet and vision transformer, for unsupervised anomaly detection and localisation. In the training phase, FastFlow learns to transform the input visual feature into a tractable distribution, and in the inference phase, it assesses the likelihood of identifying anomalies.
- [DRAEM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/draem): Is a reconstruction based algorithm that consists of a reconstructive subnetwork and a discriminative subnetwork. DRAEM is trained on simulated anomaly images, generated by augmenting normal input images from the training set with a random Perlin noise mask extracted from an unrelated source of image data. The reconstructive subnetwork is an autoencoder architecture that is trained to reconstruct the original input images from the augmented images. The reconstructive submodel is trained using a combination of L2 loss and Structural Similarity loss. The input of the discriminative subnetwork consists of the channel-wise concatenation of the (augmented) input image and the output of the reconstructive subnetwork. The output of the discriminative subnetwork is an anomaly map that contains the predicted anomaly scores for each pixel location. The discriminative subnetwork is trained using Focal Loss
- [CS-FLOW](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/csflow): The central idea of the paper is to handle fine-grained representations by incorporating global and local image context. This is done by taking multiple scales when extracting features and using a fully-convolutional normalizing flow to process the scales jointly.
- [EfficientAd](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/efficient_ad)
Fast anomaly segmentation algorithm that consists of a distilled pre-trained teacher model, a student model and an autoencoder. It detects local anomalies via the teacher-student discrepany and global anomalies via the student-autoencoder discrepancy.

For a detailed description of the models and their parameters please refer to the anomalib documentation.

For each one of these techniques there's a different config file defining the experiment foundations that can be found 
under `experiment/base/anomaly`.

#### Customizing the basic configuration
The base setup will delete the trained model at the end of the experiment to save space so be careful.

What can be useful to customize are the default callbacks:
```yaml
callbacks:
  # Anomalib specific callbacks
  score_normalization:
    _target_: quadra.utils.anomaly.ThresholdNormalizationCallback
    threshold_type: image
  post_processing_configuration:
    _target_: anomalib.utils.callbacks.post_processing_configuration.PostProcessingConfigurationCallback
    threshold_method: ${model.metrics.threshold.method}
    manual_image_threshold: ${model.metrics.threshold.manual_image}
    manual_pixel_threshold: ${model.metrics.threshold.manual_pixel}
  metrics:
    _target_: anomalib.utils.callbacks.metrics_configuration.MetricsConfigurationCallback
    task: ${model.dataset.task}
    image_metrics: ${model.metrics.image}
    pixel_metrics: ${model.metrics.pixel}
  visualizer:
    _target_: quadra.callbacks.anomalib.VisualizerCallback
    inputs_are_normalized: true
    output_path: anomaly_output
    threshold_type: ${callbacks.score_normalization.threshold_type}
    disable: true
    plot_only_wrong: false
    plot_raw_outputs: false
  batch_size_finder:
    _target_: quadra.callbacks.lightning.BatchSizeFinder
    mode: power
    steps_per_trial: 3
    init_val: 2
    max_trials: 5 # Max 64
    batch_arg_name: train_batch_size
    disable: true
```

!!! warning

    By default lightning batch_size_finder callback is disabled. This callback will automatically try to infer the maximum batch size that can be used for training without running out of memory. We've experimented runtime errors with this callback on some machines due to a Pytorch/CUDNN incompatibility so be careful when using it.

The score_normalization callback is used to normalize the anomaly maps to the range [0, 1000] such that the threshold will become 100.

The threshold_type can be either "image" or "pixel" and it indicates which threshold to use to normalize the pixel level threshold, if no masks are available for segmentation this should always be "image", otherwise the normalization will use the threshold computed without masks which would result in wrong segmentations.

The post processing configuration allow to specify the method used to compute the threshold, methods and manual metrics are generally specified in the model configuration and should not be changed here.

The visualizer callback is used to produce a visualization of the results on the test data, when the score_normalization callback is used the input_are_normalized flag must be set to true and the threshold_type should match the one used for normalization. By default it is disabled as it may take a while to compute, to enable just set `disable: false`.

In the context where many images are supplied to our model, we may be more interested in restricting the output images that are generated to only the cases where the result is not correct. By default it is disabled, to enable just set `plot_only_wrong: true`.

The display of the outputs of a model can be done in a preset format. However, this option may not be as desired, or may be affecting the resolution of the images. In order to give more flexibility to the generation of reports, the heatmap and segmentation ouput files can be generated independently and with the same resolution of the original image. By default it is disabled, to enable just set `plot_raw_outputs: true`.


### Anomalib configuration
Anomalib library doesn't use hydra but still uses yaml configurations that are found under `model/anomalib`.
This for example is the configuration used for PADIM.
```yaml
dataset:
  task: segmentation
  tiling:
    apply: false
    tile_size: null
    stride: null
    remove_border_count: 0
    use_random_tiling: False
    random_tile_count: 16

model:
  input_size: [224, 224]
  backbone: resnet18.tv_in1k
  layers:
    - layer1
    - layer2
    - layer3
  pre_trained: true
  n_features: null

metrics:
  image:
    - F1Score
    - AUROC
  pixel:
    - F1Score
    - AUROC
  threshold:
    method: adaptive # options: [adaptive, manual]
    manual_image: null
    manual_pixel: null
```
What we are mostly interested about is the `model` section. In this section we can specify the backbone of the model
(mainly resnet18.tv_in1k and wide_resnet50_2.tv_in1k), which layers are used for feature extraction and the number of features used for dimensionality reduction (there are some default values for resnet18 and wide_resnet50_2).
```
Notice: ".tv_in1k" is an extension for timm backbones' model_name which refers to torchvision pretrained weights.
```
Generally we always compute an adaptive threshold based on the validation data, but it is possible to specify a manual threshold for both image and pixel as we may want a different tradeoff between false 
positives and false negatives. The threshold specified must be the unnormalized one.

As already mentioned anomaly detection requires just good images for training, to compute the threshold used for separating good and anomalous examples it's required to have a validation set generally containing both good and anomalous examples. If the validation set is not provided the threshold will be computed on the test set.

### Experiment

Suppose that we want to run the experiment on the given dataset using the PADIM technique. We can take the generic padim config for mnist as an example found under `experiment/generic/mnist/anomaly/padim.yaml`.

```yaml
# @package _global_
defaults:
  - base/anomaly/padim
  - override /datamodule: generic/mnist/anomaly/base

export:
  types: [torchscript]

model:
  model:
    input_size: [224, 224]
    backbone: resnet18

datamodule:
  num_workers: 12
  train_batch_size: 32
  task: ${model.dataset.task}
  good_number: 9

callbacks:
  score_normalization:
    threshold_type: "image"

print_config: false

core:
  tag: "run"
  test_after_training: true
  upload_artifacts: true
  name: padim_${datamodule.good_number}_${model.model.backbone}_${trainer.max_epochs}

logger:
  mlflow:
    experiment_name: mnist-anomaly
    run_name: ${core.name}

trainer:
  devices: [0]
  max_epochs: 1
  check_val_every_n_epoch: ${trainer.max_epochs}
```

We start from the base configuration for PADIM, then we override the datamodule to use the generic mnist datamodule. Using this configuration we specify that we want to use PADIM, extracting features using the resnet18 backbone with image size 224x224, the dataset is `mnist`, we specify that the task is taken from the anomalib configuration which specify it to be segmentation. One very important thing to watch out is the `check_val_every_n_epoch` parameter. This parameter should match the number of epochs for `PADIM` and `Patchcore`, the reason is that in the validation phase the model will be fitted and we want the fit to be done only once and on all the data, increasing the max_epoch is useful when we apply data augmentation, otherwise it doesn't make a lot of sense as we would fit the model on the same, replicated data. The model will be exported at the end of the training phase, as we have specified the `export.types` parameter to `torchscript` the model will be exported only in torchscript format.

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=generic/mnist/anomaly/padim
```

## Evaluation

The same datamodule specified before can be used for inference by setting the `phase` parameter to `test`.
The dataset structure is the same as to the one used for training, but only the test folder is required.
```tree
MNIST/
├── train 
│   └── good
│       └── xyz.png
└── test
  ├── good
  │   └── xyz.png
  ├── 0
  │   └── xyz.png
  ├── 1
  │   └── xyz.png
  ├── unknown
  │   └── xyz.png
  └── ...
      └── xyz.png
```

It's possible to define a folder named `unknown` in the test folder to define images for which we don't know the label, but we still want to perform inference on them.

### Experiment

The default experiment config is found under `configs/experiment/base/anomaly/inference`.

```yaml
# @package _global_

defaults:
  - override /datamodule: base/anomaly
  - override /model: null
  - override /optimizer: null
  - override /scheduler: null
  - override /transforms: default_resize
  - override /loss: null
  - override /task: anomalib/inference
  - override /backbone: null
  - override /trainer: null
  - _self_

datamodule:
  phase: test

task:
  model_path: ???
  use_training_threshold: false
  training_threshold_type: image
```

By default, the inference will recompute the threshold based on test data to maximize the F1-score, if you want to use the threshold from the training phase you can set the `use_training_threshold` parameter to true.
The `training_threshold_type` can be used to specify which training threshold to use, it can be either `image` or `pixel`, if not specified the `image` threshold will be used.

The model path is the path to an exported model, at the moment `torchscript` and `onnx` models are supported (exported automatically after a training experiment). Right now only the `CFLOW` model is not supported for inference as it's not compatible with botyh torchscript and onnx.

An inference configuration using the mnist dataset is found under `configs/experiment/generic/mnist/anomaly/inference.yaml`.

### Run

Same as above, assuming that you have created a virtual environment and installed the `quadra` can run the experiment by running the following command:

```bash
quadra experiment=generic/mnist/anomaly/inference task.model_path={path to the exported trained model}
```

Generally for inference is enough to use the base experiment providing both model_path and data_path like this:

```bash
quadra experiment=base/anomaly/inference task.model_path={path to the exported trained model} datamodule.data_path={path to the dataset}
```