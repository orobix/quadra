<h1 align="center" style="margin-left: 20%; margin-right: 20%;">
   <svg xmlns="http://www.w3.org/2000/svg" id="Layer_1" data-name="Layer 1" viewBox="0 -16.48 1679.2 450.4">
      <defs>
         <style>
            .logo_q-color {
            fill: #009485;
            }
            .cls-1,
            .cls-2,
            .cls-3 {
            isolation: isolate;
            }
            .cls-2,
            .cls-3 {
            font-size: 406.97px;
            font-family: NeutraDisp-Bold, Neutra Display;
            font-weight: 700;
            }
            .cls-3 {
            letter-spacing: -0.04em;
            }
         </style>
      </defs>
      <g class="cls-1">
         <text class="logo_q-color cls-2" transform="translate(0 345.92)">O</text>
         <text class="logo_q-color cls-3" transform="translate(325.57 345.92)">U</text>
         <text class="logo_q-color cls-2" transform="translate(588.47 345.92)">AD</text>
         <text class="logo_q-color cls-2" transform="translate(1137.46 345.92)">R</text>
         <text class="logo_q-color cls-2" transform="translate(1385.3 345.92)">A</text>
      </g>
      <path class="logo_q-color" d="M348,314.9,300.06,280a44.56,44.56,0,0,0,9.78,62.08L357.74,377A44.55,44.55,0,0,0,348,314.9Z"
         transform="translate(-112.52 0)" />
      <path class="logo_q-color" d="M345.51,299.37l-47.9-34.86a44.58,44.58,0,0,0,9.78,62.09l47.89,34.86A44.58,44.58,0,0,0,345.51,299.37Z"
         transform="translate(-112.52 0)" />
      <path class="logo_q-color" d="M343.07,283.85,295.16,249A44.58,44.58,0,0,0,305,311.06L352.85,346A44.58,44.58,0,0,0,343.07,283.85Z"
         transform="translate(-112.52 0)" />
   </svg>
</h1>

<p align="center">
  <a href="https://orobix.github.io/quadra/latest/index.html">Docs</a> •
  <a href="https://orobix.github.io/quadra/latest/tutorials/install.html">Tutorials</a> •
  <a href="https://orobix.github.io/quadra/latest/tutorials/configurations.html">Configurations</a>
</p>

<div align="center">

  <a href="https://github.com/pre-commit/pre-commit"><img
    src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"
    alt="pre-commit"
  /></a>
  <a href="https://github.com/econchick/interrogate"><img
    src="docs/images/interrogate_badge.svg"
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
  <a href="https://github.com/psf/black"><img
    src="https://img.shields.io/badge/code%20style-black-000000.svg"
    alt="black"
  /></a>
  <a href="https://pycqa.github.io/isort/"><img
    src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"
    alt="isort"
  /></a>

</div>

______________________________________________________________________


`Quadra` aims to simplify deep learning experimenting process, allowing researchers or developers to compare, monitor, and share their experiments quickly. It provides a simple and flexible way to train and deploy deep learning models using YAML configuration files and open-source tools such as [Hydra](https://hydra.cc/docs/intro/), [Lightning](https://www.pytorchlightning.ai/index.html) framework, and [Pytorch](https://pytorch.org/). It lets you compose your experiment configurations from single command line interface, so you can conduct multiple experiments with different settings and hyperparameters. Every experiment can be logged using integrations provided by  [Lightning](https://www.pytorchlightning.ai/index.html) framework such [Mlflow](https://mlflow.org/).


## Quick Start Guide

Currently we support installing from source since the library is not yet available on `PyPI` and currently supported Python version is `3.9`.

```shell
pip install git+https://github.com/orobix/quadra.git@v1.0.0
```

If you don't have virtual environment ready, Let's set up our environment for using the `quadra` library. We have two parts in this guide: Common setup and Environment-specific setup.

### Using Conda

Create and activate a new `Conda` environment. 

```shell
conda create -n myenv python=3.9
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

3. **Install the `quadra` package**:
  ```shell
  pip install git+https://github.com/orobix/quadra.git@v1.0.0
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

BYOL is not the only SSL algorithm available, you can find a list of all the available algorithms under `quadra/experiment/generic/imagenette/ssl` folder.


### Anomaly Detection Training

To run a simple anomaly detection training on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using the PADIM algorithm run the following command:

```bash
quadra experiment=generic/mnist/anomaly/padim logger=csv
```

This will run an anomaly detection considering on of the classes as good (default is the number 9) and the rest as anomalies.

This will make use of the [anomalib](https://github.com/openvinotoolkit/anomalib) library to train the model. Many different algorithms are available, you can find them under `quadra/experiment/generic/mnist/anomaly` folder.

## Running with Custom Datasets

Each task comes with a default configuration file that can be customized for your needs. Each example experiment we have seen so far uses a default configuration file that can be found under `quadra/experiment/base/<task>/<config_name>.yaml`. 

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

For more information about advanced usage, please check [tutorials](/tutorials/configurations.html) and [task specific examples](/tutorials/examples/classification.html).

## Development

First clone the repository from Github

First clone the repository from `Github`, then we need to install the package with optional dependencies (generally in editable mode) and enable the pre-commit hooks.

1. `git clone https://github.com/orobix/quadra.git && cd quadra` 
1. Install `quadra` package in editable mode `pip install -e .[dev,test,docs]`
2. Install pre-commit hooks `pre-commit install`
3. (Optional) Eventually build documentation by calling required commands (see below).

Now you can start developing and the pre-commit hooks will run automatically to prevent you from committing code that does not pass the linting and formatting checks.

We rely on a combination of `Black`, `Pylint`, `Mypy`, `Ruff` and `Isort` to enforce code quality.

## Building Documentations

1. Activate your virtual environment.
2. Install the `quadra` package with at least `doc` version.
3. To run the webserver for real-time rendering and editing run `mkdocs serve` and visit `http://localhost:8000/`.
4. If you want to export the static website to a specific folder  `mkdocs build -d <Destination Folder>`

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

