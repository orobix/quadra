# Installation

## Virtual environment

To start a new Python environment, run the following command:

```bash
cd `<your-project-directory>`
python3 -m venv ./<environment-name>
source ./<environment-nam>/bin/activate
```

If you prefer to use `conda` instead of `venv`, run the following command:

```bash
conda create -n <environment-name> python=3.9
conda activate <environment-name>
```

!!!note
    This library relies on python version 3.8 or 3.9 at the moment.

## Installing the package

Once the environment is activated, you can install the package using the following command:

```bash
pip install .
```

!!!note
    This installs the package for deployment, there are other ways to install the package based on your needs:

    - `pip install -e .[dev]` (for development) (docs + testing)
    - `pip install -e .[docs]` (for documentation)
    - `pip install -e .[test]` (for testing)

## Usage

When you install the package, you can use the following CLI command to run the experiments:

```bash
quadra <config-composition>
```

For example this is how a simple classification experiments on the `imagenette` dataset is run:

```bash
quadra experiment=generic/imagenette/classification/classification
```

## Crendentials and Environment variables

`quadra` uses environment variables to store credentials and other sensitive information. Thanks to [`python-dotenv`](https://pypi.org/project/python-dotenv/) library, you can create a `.env` file in the main folder of your project and store the credentials there. During the runtime, the library will automatically load the environment variables from the `.env` file.

