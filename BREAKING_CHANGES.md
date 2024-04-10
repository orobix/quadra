# Breaking Changes
All the breaking changes will be documented in this file.

### [2.1.0]

#### Changed

- Change the way anomaly scores are normalized by default, instead of using a [0-1] range with a 0.5 threshold, the scores are now normalized to a [0-1000] range with a threshold of 100, the new score represents the distance from the selected threshold, for example, a score of 200 means that the anomaly score is 100% of the threshold above the threshold itself, a score of 50 means that the anomaly score is 50% of the threshold below. This change is intended to make the scores more interpretable and easier to understand, also it makes the score independent from the min and max
scores in the dataset.

### [2.0.0]

#### Changed

- Quadra 2.0.0 works with torch 2 and pytorch lightning 2, lightning trainer configurations must be aligned following the [migration guide](https://lightning.ai/docs/pytorch/LTS/upgrade/migration_guide.html).
- Quadra now relies on CUDA 12 to work instead of the old CUDA 11.6

### [1.5.0]

#### Changed

- In Quadra 1.5.0, poetry is used as the default build system for better dependency management and it should be used as primary source for development.

### [1.4.0]

#### Changed

- In Quadra 1.4.0, we upgraded timm to version 0.9.12, resulting in potential variations in default weights for timm backbones compared to previous versions. To continue utilizing the previous weights for resnet18 and wide_resnet50, which are the default backbones for quadra anomaly and classification fine-tuning tasks, we have introduced ".tv_in1k" to the model_name inside Quadra configuration files.

Although the timm upgrade might have very likely adjusted default weights also for other backbones, we have reinstated the old weights only for these two (some internal tests showed better performance of old weights, especially for classification fine-tuning).

If you are updating quadra to a version >= 1.4.0 and you want to keep consistent results, it is recommended to verify whether your timm's backbone is sill using the same weights.