# Reproducibility and Logging

One of the most important aspect of the experiment management is reproducibility which is ensured by some framework utilities.

In this page, we will introduce how experiments and configuration are stored and how they can be used in other projects. 

First of all, the configuration files contain a fixed seed under `core.seed` that is set before running any line of code. This seed is also used by `datamodule` classes.

Each experiment that has been initiated has its own folder. If the user is using the default settings. The folder will be saved under `logs/(multi)<runs>/<core.name>/YYYY-MM-DD_HH-mm-SS` that is relative to the folder where `quadra` command is running.

Typical example of the folder structure is:

```tree
logs/.../experiment_name/
├── checkpoints
│   └── ...
├── config_resolved.yaml
├── config_tree.txt
├── data
│   └── datamodule.pkl
├── deployment_model
│   └── ...
├── main.log
├── .hydra
│   ├── config.yaml
│   └── overrides.yaml
└── task_specific_outputs...
```

- **checkpoints:** This folder contains the checkpoints of the model (if any was saved).
- **config_resolved.yaml:** This file contains the resolved configuration after hydra parses it. It is useful when you want to know what is the final configuration after resolving the overrides.
- **config_tree.txt:** Human readable version of the configuration.
- **datamodule.pkl:** A serialized version of the datamodule that is used in the experiment, it contains the train, val and split datasets alongside all the other parameters.
- **deployment_model:** Contains the eventual model in deployment format.
- **main.log:** This file contains the system logs of the experiment.
- **.hydra:** This folder contains the hydra configuration file created by hydra.
- **task_specific_outputs:** These are auxiliary files that are used to store the results of the experiment, generally they are treated as artifacts so specific loggers can handle them

!!! note
        If the user specifies `core.upload_artifacts=True` in the config file, the artifacts will be uploaded to the logger artifact storage if possible.
        
!!! note
        If the user specifies `core.mlflow_zip_models=True` in the config file, the exported models will be zipped before being uploaded to mlflow, if possible.

Since all final configurations are written into `config_resolved.yaml` file, it is the single file enough to compare or replicate the experiment. However, we need to sync the codebase as the same it has been used to train the model. To do so, the quadra library also saves the git commit hash of the codebase as hyperparameter under `git/commit` tag. Using this hash, user can `git checkout <commit-hash>` the codebase and have the identical repository setup to run the same experiment.