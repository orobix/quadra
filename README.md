<h1></h1>
<p align="center">
  <img src="https://github.com/orobix/quadra/raw/main/docs/images/quadra_text_logo.png" alt="Quadra Logo" width="100%">
</p>

<p align="center">
  <a href="https://orobix.github.io/quadra/latest/index.html">Docs</a> •
  <a href="https://orobix.github.io/quadra/latest/tutorials/datamodules.html">Tutorials</a> •
  <a href="https://orobix.github.io/quadra/latest/tutorials/configurations.html">Configurations</a>
</p>

<div align="center">
  <a href="https://github.com/astral-sh/ruff">
    <img 
    src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" 
    alt="ruff"
  ></a>
  <a href="https://github.com/pre-commit/pre-commit"><img
    src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"
    alt="pre-commit"
  /></a>
  <a href="https://github.com/econchick/interrogate"><img
    src="https://github.com/orobix/quadra/raw/main/docs/images/interrogate_badge.svg"
    alt="interrogate"
  /></a>
  <a href="https://github.com/PyCQA/pylint"><img
    src="https://img.shields.io/badge/linting-pylint-yellowgreen"
    alt="pylint"
  /></a>
  <a href="https://mypy-lang.org/"><img
    src="https://www.mypy-lang.org/static/mypy_badge.svg"
    alt="Checked with mypy"
  /></a>

</div>

______________________________________________________________________


`Quadra` aims to simplify deep learning experimenting process, allowing researchers or developers to compare, monitor, and share their experiments quickly. It provides a simple and flexible way to train and deploy deep learning models using YAML configuration files and open-source tools such as [Hydra](https://hydra.cc/docs/intro/), [Lightning](https://www.pytorchlightning.ai/index.html) framework, and [Pytorch](https://pytorch.org/). It lets you compose your experiment configurations from single command line interface, so you can conduct multiple experiments with different settings and hyperparameters. Every experiment can be logged using integrations provided by  [Lightning](https://www.pytorchlightning.ai/index.html) framework such [Mlflow](https://mlflow.org/).


## Quick Start Guide

If you use pip to manage your packages, you can install `quadra` from PyPi by running the following command:
```shell
pip install quadra
```

If instead you prefer to use poetry, you can install `quadra` from PyPi by running the following command:
```shell
poetry add quadra
```

If you don't have virtual environment ready, Let's set up our environment for using the `quadra` library. We have two parts in this guide: Common setup and Environment-specific setup.

### Using Conda

Create and activate a new `Conda` environment. 

```shell
conda create -n myenv python=3.10
conda activate myenv
```

### Using Python Virtualenv

Create and activate a new virtual environment.

```shell
# replace `myenv` with the name of your virtual environment
python3 -m venv myenv
source myenv/bin/activate
```

### Common Setup

1. **Check your git version**: 
  Make sure you have git version 2.10 or higher, to avoid any installation failures.
  ```shell
  git --version
  ```

2. **Upgrade pip**:
  ```shell
  pip install --upgrade pip
  ```

3. Install the package

    * **Install the `quadra` package** with pip:
      ```shell
      pip install quadra
      ```

    * **Install the `quadra` package** with poetry:
      ```shell
      curl -sSL https://install.python-poetry.org | python3 -
      poetry add quadra
      ```

4. **Run from CLI**:
  Run the following command to check if the installation was successful:
  ```shell
  quadra experiment=default
  ```

### Setup `Mlflow` (Optional)

To use Mlflow and leverage its functionalities such as saving models, logging metrics, saving artifacts, and visualizing results, you need to ensure that the Mlflow server is running. You can find more information about Mlflow [here](https://mlflow.org/).

By default, the logger configuration is set to Mlflow, and experiments expect the `MLFLOW_TRACKING_URI` environment variable to be set to the address of the Mlflow server. There are two ways to set this variable:

**Using the command line:**

```
export MLFLOW_TRACKING_URI=http://localhost:5000
```
This command sets the `MLFLOW_TRACKING_URI` variable to `http://localhost:5000`. Replace this with the actual address of your Mlflow server if it's running on a different host or port.

**Adding it to your environment file:**

`Quadra` uses environment variables to store credentials and other sensitive information. Thanks to [`python-dotenv`](https://pypi.org/project/python-dotenv/) library, you can create a `.env` file in the main folder of your project and store the credentials there. During the runtime, the library will automatically load the environment variables from the `.env` file. You can also add the `MLFLOW_TRACKING_URI` variable to your environment file (e.g., `.env`). Open the file in a text editor and add the following line:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```
Again, modify the address if your Mlflow server is running on a different host or port.

By setting the `MLFLOW_TRACKING_URI` variable using either method, you configure the logger to connect to the Mlflow server, enabling you to utilize its features effectively.

The `export` command is specific to Unix-based systems like Linux or macOS. If you are using a different operating system, refer to the appropriate method for setting environment variables.

## Run Example Experiments

`quadra` provides a set of example experiments that can be used to test the installation and to provide some example configuration files.

By default all the experiments will run on the `GPU 0`, to run it on the `CPU` you can specify a different `trainer` configuration parameter:

```bash
quadra <configurations> trainer=lightning_cpu
```

### Classification Training

To run a simple classification training on the [Imagenette dataset](https://github.com/fastai/imagenette) with a Resnet18 architecture run the following command:

```bash
quadra experiment=generic/imagenette/classification/default logger=csv
```

This will train the model for 20 epochs and log the metrics in a csv file, at the end of the training a `torchscript` model will be saved for inference alongside some output images.

By default the experiment will run on the GPU 0, to run it on the CPU you can specify a different `trainer` configuration parameter.


### Segmentation Training

To run a simple segmentation training on the [Oxford pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset with a Unet architecture run the following command:

```bash
quadra experiment=generic/oxford_pet/segmentation/smp logger=csv
```

This will make use of the [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) library to train the model for 10 epochs, logging the metrics to a csv file. At the end of the training a torchscript model will be saved for inference alongside some output images.

### (SSL) Self-supervised Learning Training

On the same dataset we can run a simple SSL training using the BYOL algorithm with the following command:

```bash
quadra experiment=generic/imagenette/ssl/byol logger=csv
```

BYOL is not the only SSL algorithm available, you can find a list of all the available algorithms under `quadra/configs/experiment/generic/imagenette/ssl` folder.


### Anomaly Detection Training

To run a simple anomaly detection training on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using the PADIM algorithm run the following command:

```bash
quadra experiment=generic/mnist/anomaly/padim logger=csv
```

This will run an anomaly detection considering on of the classes as good (default is the number 9) and the rest as anomalies.

This will make use of the [anomalib](https://github.com/openvinotoolkit/anomalib) library to train the model. Many different algorithms are available, you can find them under `quadra/configs/experiment/generic/mnist/anomaly` folder.

## Running with Custom Datasets

Each task comes with a default configuration file that can be customized for your needs. Each example experiment we have seen so far uses a default configuration file that can be found under `quadra/configs/experiment/base/<task>/<config_name>.yaml`. 

Let's see how we can customize the configuration file to run the classification experiment on a custom dataset.

Structure your dataset in the following way:

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
```

Create a new experiment configuration file under `quadra/configs/experiment/custom/<config_name>.yaml` with the following content:

```yaml
# @package _global_
defaults:
  - base/classification/classification # extends the default classification configuration

core:
  name: <your-custom-experiment-name> # name of the experiment

model:
  num_classes: <number-of_classes-you-have> # number of classes in your dataset
```

Run the experiment with the following command:

```bash
quadra experiment=custom/<config_name> logger=csv
```

It will run the experiment using the configuration file you have just created and it will apply the default parameters from the classification configuration file. Furthermore, it will log the metrics to a csv file. You can add or customize the parameters in the configuration file to fit your needs.

For more information about advanced usage, please check [tutorials](https://orobix.github.io/quadra/latest/tutorials/configurations.html) and [task specific examples](https://orobix.github.io/quadra/latest/tutorials/examples/classification.html).

## Development

First clone the repository from Github

First clone the repository from `Github`, then we need to install the package with optional dependencies (generally in editable mode) and enable the pre-commit hooks.

1. `git clone https://github.com/orobix/quadra.git && cd quadra` 
2. Install poetry `curl -sSL https://install.python-poetry.org | python3 -`
3. Install the required poetry plugins
  ```
  poetry self add poetry-bumpversion
  poetry self add poetry-dotenv-plugin
  ```
4. Install `quadra` package in editable mode `poetry install --with test,dev,docs --all-extras`
5. Install pre-commit hooks `pre-commit install`
6. (Optional) Eventually build documentation by calling required commands (see below).

Now you can start developing and the pre-commit hooks will run automatically to prevent you from committing code that does not pass the linting and formatting checks.

We rely on a combination of `Pylint`, `Mypy` and `Ruff` to enforce code quality.

## Building Documentations

1. Activate your virtual environment.
2. Install the `quadra` package with at least `doc` version.
3. To run the webserver for real-time rendering and editing run `mkdocs serve` and visit `http://localhost:8000/`.
4. If you want to export the static website to a specific folder  `mkdocs build -d <Destination Folder>`


## Acknowledgements

This project is based on many open-source libraries and frameworks, we would like to thank all the contributors for their work. Here is a list of the main libraries and frameworks we use:

- [Pytorch](https://pytorch.org/) and [Pytorch Lightning](https://lightning.ai/) for training and deploying deep learning models. These two libraries are core part of training and testing tasks that allow us to run experiments on different devices in agile way.
- Pretrained models are usually loaded from [Pytorch Hub](https://pytorch.org/hub/) or [Pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (or called as `timm`).
- Each specific task may rely on different libraries. For example, `segmentation` task uses [Segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) for loading backbones. The `anomaly detection` task uses a fork of [Anomalib](https://github.com/openvinotoolkit/anomalib) maintained by Orobix on [this repository](https://github.com/orobix/anomalib). We use light-weight ML models from [scikit-learn](https://scikit-learn.org/). We have also implementation of some SOTA models inside our library. 
- Data processing and augmentation are done using [Albumentations](https://albumentations.ai/docs/) and [OpenCV](https://opencv.org/).
- [Hydra](https://hydra.cc/docs/intro/) for composing configurations and running experiments. Hydra is a powerful framework that allows us to compose configurations from command line interface and run multiple experiments with different settings and hyperparameters. We have followed suggestions from `Configuring Experiments` section of [Hydra documentation](https://hydra.cc/docs/patterns/configuring_experiments/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) repository.
- Documentation website is using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and [MkDocs](https://www.mkdocs.org/). For code documentation we are using [Mkdocstrings](https://mkdocstrings.github.io/). For releasing software versions we combine [Bumpver](https://github.com/mbarkhau/bumpver) and [Mike](https://github.com/jimporter/mike).
- Models can be exported in different ways (`torchscript` or `torch` file). We have also added [ONNX](https://onnx.ai/) support for some models.
- Testing framework is based on [Pytest](https://docs.pytest.org/en/) and related plug-ins.
- Code quality is ensured by [pre-commit](https://pre-commit.com/) hooks. We are using [Ruff](https://github.com/astral-sh/ruff) for linting, enforcing code quality and formatting, [Pylint](https://www.pylint.org/) for in depth linting and [Mypy](https://mypy.readthedocs.io/en/stable/) for type checking.

## FAQ

**How can I fix errors related to `GL` when I install full `opencv` package?**

If you are running on a remote server without a display and you are using `opencv-python` instead of `opencv-python-headless` you can run the following command to fix the issue:

It can be solved by correctly linking the `libGL.so.1` library:

```bash
# Check where the library is located
find /usr -name libGL.so.1
```

```bash
# if the library is located in /usr/lib/x86_64-linux-gnu
# link it to /usr/lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/libGL.so
```

or installing following packages (depending on your OS this may vary):

```bash
sudo apt-get install libgl1-mesa-glx
```


**How can I run multiple experiments with single command?**

You can run multiple experiments with a single command by passing `--multirun` flag:

```bash
quadra <configurations> --multirun
```

For example if you want to run same experiment with different seeds you can run:

```bash
quadra experiment=generic/imagenette/classification/default trainer=lightning_cpu logger=csv core.seed=1,2,3 --multirun 
```

