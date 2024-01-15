# Breaking Changes
All the breaking changes will be documented in this file.

### [1.5.0]

#### Changed

- In Quadra 1.5.0, poetry is used as the default build system for better dependency management and it should be used as primary source for development.

### [1.4.0]

#### Changed

- In Quadra 1.4.0, we upgraded timm to version 0.9.12, resulting in potential variations in default weights for timm backbones compared to previous versions. To continue utilizing the previous weights for resnet18 and wide_resnet50, which are the default backbones for quadra anomaly and classification fine-tuning tasks, we have introduced ".tv_in1k" to the model_name inside Quadra configuration files.

Although the timm upgrade might have very likely adjusted default weights also for other backbones, we have reinstated the old weights only for these two (some internal tests showed better performance of old weights, especially for classification fine-tuning).

If you are updating quadra to a version >= 1.4.0 and you want to keep consistent results, it is recommended to verify whether your timm's backbone is sill using the same weights.