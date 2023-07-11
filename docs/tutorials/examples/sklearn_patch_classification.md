# Sklearn patch classification example

In this page, we will show you how to train an Sklearn classifier using a Pytorch feature extractor in a patch based fashion.

This example will demonstrate how to create custom experiments starting from default settings.

## Training
### Dataset

Let's start with the dataset that we are going to use; since we are using the base patch datamodule, we need to organize the data in a specific way. The base patch datamodule expects the following structure:

```tree
patch_dataset/
├── info.json
├── original 
│   ├── images
│   |   └── abc.xyz
│   ├── masks
|   |   └── abc.png
|   └── labelled_masks
|       └── abc.png
├── train
│   ├── abc_0.h5
│   ├── abc_N.h5
│   └── dataset.txt
├── val
│   ├── class_0
│   |   └── abc_X.xyz
│   ├── class_1
|   |   └── abc_Y.xyz
|   └── class_N
|       └── abc_Z.xyz
└── test
    ├── class_0
    |   └── abc_X.xyz
    ├── class_1
    |   └── abc_Y.xyz
    └── class_N
        └── abc_Z.xyz
```

To achieve this structure we need to use two different functions taken from `quadra` utilities.

- `quadra.utils.patch.get_image_mask_association`: This function will create a list of dictionaries mapping images and masks (if available).
- `quadra.utils.patch.generate_patch_dataset`: This function will generate the dataset following the structure above.

For example imagine we have a standard dataset with the following structure:

```tree
dataset/
├── images
|   ├── abc.xyz
│   └── ...
└── masks
    ├── abc.png
    └── ...
```

By calling `get_image_mask_association` passing the right data_folder and mask_folder, we will get the following dictionary:

```json
[
  {
      "base_name": "abc.png",
      "path": "dataset/images/abc.png",
      "mask": "dataset/masks/abc.png"
  }, ...
]
```

The function allows also to specify an extension name for masks (E.g. _mask) for the mapping, for more details see the function documentation.
This is just an helper, you can create the dictionary in any way you want, but the dictionary must follow this structure so that the `generate_patch_dataset` function can work.

!!! note
    Masks contain the labels of the images in index format. For example, if the image has 3 classes, the mask will contain values between 0 and 2. The `class_to_idx` parameter of the `generate_patch_dataset` function will be used to map the index to the class name. 0 will be considered as the background class, as such it is generally ignored when making predictions. 

Now that we have the dictionary, we can call `generate_patch_dataset` to generate the dataset. Here it's an example:

```python
generate_patch_dataset(
    data_dictionary: my_dictionary,
    class_to_idx: {"background": 0, "class_1": 1, "class_2": 2},
    val_size: 0.3,
    test_size: 0.1,
    seed: 42,
    patch_number: [16, 16],
    patch_size: None, # Either patch_size or patch_number must be specified
    overlap: 0.0,
    save_original_images_and_masks: bool = True,
    output_folder: "patch_dataset",
    area_threshold: 0.45,
    area_defect_threshold: 0.2,
    mask_extension: "_mask",
    mask_output_folder: None,
    save_mask: False,
    clear_output_folder: False,
    mask_preprocessing: None,
    train_filename: "dataset.txt",
    repeat_good_images: 1,
    balance_defects: True,
    annotated_good: None,
    num_workers: 1,
) 
```

This function will generate the dataset in the specified output folder. The function has a lot of parameters, but the most important are:

- `data_dictionary`: The dictionary containing the mapping between images and masks.
- `class_to_idx`: A dictionary mapping the class names to the corresponding index.
- `patch_number`: Specify the number of [vertical, horizontal] patches to extract from each image. If the image is not divisible by the patch number the patch size and overlap will be adjusted to fit the image as much as possible, if some part of the image is not covered properly on the edges that part will be taken from the border of the image and the reconstruction will be done accordingly.
- `patch_size`: Specify the size [h, w] of the patches to extract from each image. If this is specified, the `patch_number` parameter will be ignored.
- `overlap`: The overlap between patches. This is a float between 0 and 1.
- `area_threshold`: The minimum percentage area of the patch that must be covered by an annotation to be considered of that class. For example if we have 224x224 patches and an area_threhsold of 0.2, the patch will be considered as a specific class if the area of an annotation overlapping that patch it at least (224 * 224 * 0.2) pixels, otherwise it will be considered as background.
- `area_defect_threshold`: The minimum area of an annotation to consider the patch as a defect. This is useful when you have small annotations that will always be considered as good patches by the `area_threshold` parameter. For example if we have 224x224 patches and an area_defect_threshold of 0.2, even if the area overlapping is smaller than the one specified by the `area_threshold` parameter, the patch will be considered as a defect if the area of the annotation itself is at least the 20% of the totoal area of the annotation.
- `repeat_good_images`: The number of times good h5 will be repeated in the train txt.
- `balance_defects`: If True, the number of good and bad patches will be balanced in the train txt.
- `annotated_good`: Specify the class of the good patches. If None, all the patches with an annotation will be considered as defected patches and only the index 0 will be considered as good.
- `save_original_images_and_masks`: By default images and masks will be copied in the new dataset folder so that they can be used in the future even if the original dataset is deleted or moved. If this is set to False, the original images and masks will not be copied and the h5 files for training will contain a reference to the original images and masks.

The mask related parameters are used to save on disk also masks for potential patch based segmentation, but they are not used for this part.
Eventually it's possible to specify a `mask_preprocessing` function that will be applied to the masks before extracting patches (for example to convert the masks to a binary format).

This function will generate multiple subfolders in the output folder:

- `original`: This folder will contain the original images and masks (if available) if the `save_original_images_and_masks` parameter is set to True.
- `train`: This folder will contain the train h5 files and the train txt file.
- `val`: This folder will contain the validation images in the standard classification split (based on label).
- `test`: This folder will contain the test images in the standard classification split (based on label).

To create the training dataset each single annotation is converted into a polygon, the polygon is divided into triangles using [polygon triangulation](https://en.wikipedia.org/wiki/Polygon_triangulation), this allow for sampling uniformly center points for patches extraction (see [this example](https://blogs.sas.com/content/iml/2020/10/21/random-points-in-polygon.html)), triangles and their sample probability are saved inside h5 files alongside information about the original image and the annotation. The h5 files are then used to sample patches during training.
A txt file (generally named `dataset.txt`) is also generated and contains the list of h5 files to use for training. By regulating the replication of good images and the balancing of good and bad patches, it's possible to control the ratio of good and bad patches in the training dataset.

Validation and test datasets are generated by simply splitting the original images into patches of the specified size and saving them in the corresponding folder based on the label. 

During validation, test and inference a prediction is made for each patch and the final prediction is obtained by aggregating the predictions of the patches. There are currently two ways to aggregate the predictions:

- `major_voting`: The pixel prediction will be the class with the highest number of votes in all the patches that overlap that pixel, in case of tie the order is established by the class index.
- `priority`: The pixel prediction is determined by the class of the patch with the highest priority. The priority is determined by the order of the classes in the `class_to_idx` from high to low.

Finally a `info.json` file is generated and saved in the output folder. This file contains information about the dataset, like the number of patches, which annotated_good classes were used, and the list of basenames for train, val and test.
This file is loaded by the datamodule and used for different task operations.

The standard datamodule configuration for patch training is found under `datamodule/base/sklearn_classification_patch.yaml`.

```yaml
_target_: quadra.datamodules.PatchSklearnClassificationDataModule
data_path: path_to_patch_dataset
train_filename: dataset.txt
exclude_filter:
include_filter:
class_to_idx: this should be the same as the one used to generate the dataset
seed: 42
batch_size: 32
num_workers: 8
train_transform: ${transforms.train_transform}
test_transform: ${transforms.test_transform}
val_transform: ${transforms.val_transform}
balance_classes: false
class_to_skip_training:
```

If `balance_classes` is set to true, the classes will be balanced by randomly replicating samples for the less frequent classes. This is useful when the dataset is imbalanced.
`class_to_skip_training` can be used to specify a list of classes that will be excluded from training. This classes will be also ignored during validation.
This is particularly useful to skip the background class (generally index 0) when the image is just partially annotated and the background may contain actual defects.

### Experiment

Suppose that we want to run the experiment on the given dataset, we can define a config starting from the base config:
```yaml
# @package _global_

defaults:
  - override /model: logistic_regression
  - override /transforms: default_resize
  - override /task: sklearn_classification_patch
  - override /backbone: dino_vitb8
  - override /trainer: sklearn_classification
  - override /datamodule: base/sklearn_classification_patch

backbone:
  model:
    pretrained: true
    freeze: true

core:
  tag: "run"
  name: "sklearn-classification-patch"

trainer:
  iteration_over_training: 20 # Regulate how many patches are extracted

datamodule:
  num_workers: 8
  batch_size: 32
```

By default the experiment will use dino_vitb8 as backbone, resizing the images to 224x224 and training a logistic regression classifier. Patches are extracted iterating 20 times over the training dataset (since sampling is random) to get more information.

An actual configuration file based on the above could be this one (suppose it's saved under `configs/experiment/custom_experiment/sklearn_classification_patch.yaml`):
```yaml
# @package _global_

defaults:
  - base/classification/sklearn_classification_patch
  - override /backbone: resnet18
  - _self_

core:
  name: experiment-name

datamodule:
  data_path: path_to_patch_dataset
  batch_size: 256
  class_to_idx:
    background: 0
    class_1: 1
    class_2: 2
  class_to_skip_training:
    - background

task:
  device: cuda:2
  output:
    folder: classification_patch_experiment
    export_type: [torchscript]
    save_backbone: false
    report: true
    example: true
    reconstruction_method: major_voting
```

This will train a resnet18 model on the given dataset, using 256 as batch size and skipping the background class during training.
The experiment results will be saved under the `classification_patch_experiment` folder. The deployment model will be generated but only the classifier will be saved (in joblib format), to reconstruct patches for evaluation the `major_voting` method will be used.

### Run

Assuming that you have created a virtual environment and installed the `quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/sklearn_classification_patch
```

This will run the experiment training a classifier and validating on the validation dataset. Patch based and reconstruction based metrics will be computed and saved under the task output folder. The output folder should contain the following files:

```bash
classification_patch_experiment  data              reconstruction_results.json
config_resolved.yaml             deployment_model
config_tree.txt                  main.log
```

Inside the `classification_patch_experiment` folder you should find some report utilities computed over the validation dataset, like the confusion matrix. The `reconstruction_results.json` file contains the reconstruction metrics computed over the validation dataset in terms of covered defects, it will also contain the coordinates of the polygons extracted over predicted areas of the image with the same label.

The `data` folder contains a joblib version of the datamodule containing parameters and splits for reproducibility. The `deployment_model` folder contains the backbone exported in torchscript format if `save_backbone` to true alongside the joblib version of trained classifier.

## Evaluation
The same datamodule specified before can be used for inference. 

### Experiment

The default experiment config is found under `configs/experiment/base/classification/sklearn_classification_patch_test.yaml`.

```yaml
# @package _global_

defaults:
  - override /transforms: default_resize
  - override /task: sklearn_classification_patch_test
  - override /datamodule: base/sklearn_classification_patch
  - override /trainer: sklearn_classification

core:
  tag: "run"
  name: "sklearn-classification-patch-test"

datamodule:
  num_workers: 8
  batch_size: 32
```

Here we specify again the backbone as it's required if the runtime model was generated without saving it in a deployment format.

An actual configuration file based on the above could be this one (suppose it's saved under `configs/experiment/custom_experiment/sklearn_classification_patch_test.yaml`):
```yaml
# @package _global_
defaults:
  - base/classification/sklearn_classification_patch_test
  - override /backbone: resnet18
  - _self_

core:
  name: experiment-test-name

datamodule:
  data_path: path_to_patch_dataset
  batch_size: 256

task:
  device: cuda:2 # Specify the device to use
  output:
    folder: classification_patch_test
    report: true
    example: true
    reconstruction_method: major_voting
  model_path: ???
```

This will test the model trained in the given experiment on the given dataset. The experiment results will be saved under the `classification_patch_test` folder.
Patch reconstruction will be performed using the `major_voting` method (can be this or `priority`). The `model_path` parameter is required to specify the path to the trained model. It could either be a '.pt'/'.pth' or a backbone_config '.yaml' file. In this case is not necessary to specify `class_to_idx` and `class_to_skip_training` as they will be loaded from the training experiment.

### Run

Same as above, assuming that you have created a virtual environment and installed the 
`quadra` library, you can run the experiment by running the following command:

```bash
quadra experiment=custom_experiment/sklearn_classification_patch_test
```