
# Changelog
All notable changes to this project will be documented in this file.

### [1.1.4]

#### Fixed

- Fix input shape not extracted properly in ModelSignatureWrapper when wrapping around a torchscript model.
### [1.1.3]

#### Fixed

- Fix penultimate patch extracted containing replicated data, now the penultimate patch contain the right data. Fix also the patch size and step computation to avoid generating more than one extra patch.

### [1.1.2]

#### Fixed

- Fix best checkpoint not used for testing when available in `LightningTask` class.

### [1.1.1]

#### Fixed

- Fix deprecated link for tutorials in `README.md` file.

### [1.1.0]

#### Added

- Add ModelManager class to manage model deployments on model tracking platforms (e.g. MLFlow).
- Add automatic storage of file hashes in BaseDataModule class for better experiment reproducibility and tracking.
- Automatically load transforms parameters from model info file in base Evaluation task.
- Add support for vit explainability using Attention Gradient Rollout method.
- Add export of pytorch model for Classification and SklearnClassification tasks.
- Add automatical detection of model input shapes for better exportation capabilities. Add support for custom input shapes like models with multiple inputs.
- Add documentation landing page, improve colore themes and logo.
- Add github actions to automatically build and deploy documentation of main and dev PRs.

#### Changed

- Refactor evaluation task to be more generic and improve inheritance capabilities.
- Refactor export_types parameter to export for better configurability of export parameters.
- Change input_size model info parameter from HXW to a list of actual model parameters for inference (e.g [(3, 224, 224)]).

#### Fixed

- Fix gradcam not working properly with non rectangular images.
- Fix logistic regression wrapper not working properly with 2 classes for torch classification.
- Fix wrong typings in NetworkBuilder's init method.
- Fix broken links in documentation.
- Fix minor documentations issues.

### [1.0.2]

#### Fixed

- Fix anomaly detection training not working when images with non lowercase allowed extension are used (e.g. .BMP).

#### Added

- Increase the number of available extensions for classification and anomaly detection training.

### [1.0.1]

#### Fixed

- Fix training dataset used instead of validation dataset in segmentation datamodules.

### [1.0.0]

#### Added

- All required files for the first release.
