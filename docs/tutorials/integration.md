# Integration with external projects

This library is designed to be used as a starting point for machine learning projects.
Imagine you want to train a classification model on your datasets, but you don't want to spend time rewriting the code.
If quadra is installed, you have access to all the configuration defined in the repository and calling

```bash
quadra {overrides}
```

Will run the experiment with the given overrides.

If you want to add more hydra configurations on your own and make
them available to quadra, it's necessary to create a `.env` file in the folder you are working on and specify the
path or the package where to find the configuration as `QUADRA_SEARCH_PATH` variable.

Config specification must follow the schema defined by [hydra config search path](https://hydra.cc/docs/advanced/search_path/).
Multiple configs can be specified by separating them with a semicolon.

For example:

```bash
QUADRA_SEARCH_PATH=file://configs;pkg://mypackage.configs
```

!!! warning
    Be careful that the configs share the same "space" of the quadra one, so it's required to avoid name collisions
    to avoid errors. One easy way to do so is to wrap the configs under a subfolder with the name of the project.
    E.g. `configs/datamodule/myproject/myconfig.yaml`

!!! warning
    If you have installed your configs inside a package you need to make sure that the package actually contains them, by default yaml files are not packaged!
    One clean way to solve this issue is to create a MANIFEST.in file in your repository containing a line like this one:
    ```
    recursive-include your_package_name *.yaml
    ```
    And set the `include_package_data` flag to `True` in your `setup.py` file.

## Debugging with VSCode

If you are developing an external project which uses `quadra` as experiment manager, you can setup debugger configurations to attach debugger to the training process.

In VSCode, you can create a `launch.json` file in the `.vscode` folder of your project. The following is an example of a configuration that can be used to debug the training process of a segmentation model.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "segmentation",
            "type": "python",
            "request": "launch",
            "program": "quadra.main",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "HYDRA_FULL_ERROR": "1",
            },
            "args": [ 
                "experiment=generic/oxford_pet/segmentation/smp.yaml",
                "trainer.max_epochs=1",
                "trainer.devices=1",
                "trainer.accelerator=cpu"
            ]
        },
    ]
}
```

## Debugging with PyCharm

Debugging with PyCharm is very similar to vscode, we can make use of the GUI to pick `quadra.main` as the entry point and pass the arguments as a string.

<p align="center">
    <figure> 
        <img src="..\images\pycharm_debugger.png" title="Selecting Suitable Pytorch Version"> 
        <figcaption>Example of a valid debug configuration</figcaption> 
    </figure>
</p>