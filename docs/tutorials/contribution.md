# Contribution

In this guide, we'll cover the steps you should take to ensure your code meets the following standards:

- Pre-commit hooks for formatting code and checking for common errors.
- Google-style docstrings for documenting source code.
- Type annotations for functions and variables using `mypy`.
- Unit tests using `pytest` to ensure code correctness.

## Setting up the Environment

Before contributing to the repository, you'll need to set up your development environment. Please check the [Getting Started Guide](../getting_started.md) for instructions on how to set up your environment.

After setting up your environment you can install `Quadra` Library in different ways:

!!!info

    - `poetry install -E dev` (for development) 
    - `pip install -E docs` (for documentation)
    - `pip install -E test` (for testing)
    

## Pre-commit Hooks

Pre-commit hooks are scripts that run before a commit is made to the repository. They can be used to check for common errors, enforce code formatting standards, and perform other tasks. The repository should have a pre-commit configuration file already set up. To use it, run the following command:

```
pre-commit install
```

This will install the pre-commit hooks and they will run automatically before each commit. The pre-commit hooks will check the code for formatting errors using many built-in library defined under `.pre-commit-config.yaml` file. If any errors are found, the commit will fail and you will need to fix the errors before you can commit your changes.

## Google-style Docstrings

This library is using `Google-style` docstrings. They provide a consistent format for documenting functions and classes, making it easier for other developers to understand how to use the code. Here's an example of a Google-style docstring:

```python
def my_function(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.
    """
```

To ensure that all functions and classes in the repository are documented with Google-style docstrings, we are using [ruff](https://github.com/charliermarsh/ruff) as a pre-commit hook that checks for missing or incorrect docstrings.

## Type Annotations with Mypy

Type annotations are a way to specify the expected types of function arguments and return values. This makes it easier to understand how functions should be used and can catch type-related errors early. To enforce type annotations, we use `mypy`, a static type checker for Python. It is installed as a pre-commit hook, so it will run automatically before each commit. We don't enforce every rule in `mypy`, but we do require that all functions and variables have type annotations. Please check the [mypy documentation](https://mypy.readthedocs.io/en/stable/) for more information on how to use type annotations. If you are interested in which rules we enforce, you can check the `pyproject.toml` file in the root of the repository.

## Unit Tests with Pytest

Unit tests are a critical part of ensuring code correctness. They should be written to test each function and class in the repository, verifying that they behave as expected under different conditions. We use `pytest` to run the unit tests. We have a `tests` folder in the root of the repository that contains all the unit tests. Under this folder, there is a `conftest.py` file that contains the fixtures used by the unit tests. The fixtures are used to set up the environment for each test, and they are automatically run before each test. Here is the test folder structure:

```tree
tests/
├── configurations
├── conftest.py
├── datamodules
├── datasets
├── models
└── tasks
```

## Adding Something New

Here are the usual steps you should take when adding something new to the repository:

1. Create a new branch for your changes.
2. Add task, model, datamodule or other type of component under the package folder.
3. Add relevant configuration files under the `configs` folder.
4. Add unit tests under the `tests` folder.
5. Add documentation under the `docs` folder.
6. Run the pre-commit hooks to check for errors.
7. Open a pull request to merge your changes into the main or development branch.
8. Assign maintainers to review your pull request.
