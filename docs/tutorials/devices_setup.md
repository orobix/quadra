# Set up devices

In this section we will see how to set up the devices for training and inference.

### Lightning based tasks

If you are using a Lightning based task, you can select the appropriate `trainer` configuration based on your needs. By default experiments will use the `lightning_gpu` configuration which is designed specifically for single gpu training. The configuration file is located at `configs/trainer/lightning_gpu.yaml` and it is shown below:

```yaml
_target_: pytorch_lightning.Trainer
devices: [0]
accelerator: gpu
min_epochs: 1
max_epochs: 10
resume_from_checkpoint: null
log_every_n_steps: 10
```

We provide also configurations for multi-gpu training (`lightning_multigpu.yaml`) and cpu training (`lightning_cpu.yaml`). You can also create your own configuration to use different accelerators.

It's important that in the final experiment configuration you set the `trainer` key to the name of the configuration you want to use. For example:

```yaml
defaults:
  - base/classification/classification
  - override /trainer: lightning_multigpu

trainer:
  devices: [0, 1] # Use two gpus
```

### Sklearn based tasks

For Sklearn based tasks there's generally a `device` field in the configuration file. For example, in the `configs/task/sklearn_classification.yaml` file we have:

```yaml
_target_: quadra.tasks.SklearnClassification
device: "cuda:0"
output:
  folder: "classification_experiment"
  save_backbone: false
  report: true
  example: true
  test_full_data: true
export_config:
  types: [torchscript]
  input_shapes: # Redefine the input shape if not automatically inferred
```

You can change the device to `cpu` or a different cuda device depending on your needs.




