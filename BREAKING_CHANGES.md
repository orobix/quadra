# Breaking Changes
All the breaking changes will be documented in this file.

### [1.4.0]

#### Changed

- Starting from version 1.4.0, default weights for "resnet18" and "wideresnet_50" may vary from prior quadra versions due to the timm upgrade to version 0.9.12. To continue using the old torchvision weights for resnet18 and wideresnet, specify ".tv_in1k," which has already been set as the default in the model configurations. While this is applicable to any backbone, it's particularly emphasized for these two, as they are default choices for anomaly and classification fine-tuning tasks. If updating quadra to a version >= 1.4.0, it's advisable to verify if your timm's backbone maintains consistent weights.