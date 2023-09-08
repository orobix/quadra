# Export models for inference

In this section we will see how quadra allows you to export your trained models for inference. We will see how to export models for both Lightning based tasks and Sklearn based tasks.
By default the standard export format is Torchscript, but you can also export models to ONNX or plain Pytorch (this is done for particular operations like gradcam).

## Standard export configuration

The standard configuration for exporting models is located at `configs/export/default.yaml` and it is shown below:

```yaml
types: [torchscript]
input_shapes: # Redefine the input shape if not automatically inferred
onnx:
  # torch.onnx.export options
  input_names: # If null automatically inferred
  output_names: # If null automatically inferred
  dynamic_axes: # If null automatically inferred
  export_params: true
  opset_version: 16
  do_constant_folding: true
  # Custom options
  fixed_batch_size: # If not null export with fixed batch size (ignore dynamic axes)
  simplify: true
```

`types` is a list of the export types that you want to perform. The available types are `torchscript`, `onnx` and `pytorch`. By default the models will be saved under the `deployment_model` folder of the experiment with extension `.pt`, `.onnx` and `.pth` respectively. Pytorch models will be saved alongside the yaml configuration for the model itself so that you can easily load them back in python.

`input_shapes` is a parameter that will be `None` most of the time, quadra features a model wrapper that is capable of inferring the input shape of the trained model based on its forward function. It supports a large variety of custom forward functions where parameters are combinations of lists, tuples or dicts. However, if the model wrapper is not able to infer the input shape you can specify it here, the format is a list of tuples/lists/dicts where each element represents a single input shape without batch size. For example if your model has an input of shape (1, 3, 224, 224) you can specify it as:

```yaml
input_shapes:
  - [3, 224, 224]
```

Onnx support a set of extra options passed as kwargs to the `torch.onnx.export`, once again we will try to automatically infer most of them but if you need to specify them you can do it under the `onnx` section of the configuration file. The two custom options are:

- `fixed_batch_size`: It can be used to export the model with a fixed batch size, this is useful if you want to use the model in a context where you know the batch size in advance. If you specify this option the model will be exported with a fixed batch size and the dynamic axes will be ignored.
- `simplify`: If true the model will be simplified using the [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) package, the resulting model is called `model_simplified.onnx` and it is saved alongside the original model.

## Lightning based tasks

Currently quadra supports exporting models for the following tasks:

- Image classification
- Image segmentation
- Anomaly detection (certain models may not be supported)
- SSL training

## Sklearn based tasks

Currently quadra supports exporting models for the following tasks:

- Image classification
- Image classification with patches

When working with sklearn based tasks alongside the exported backbone in the `deployment_model` folder you will also find a `classifier.joblib` containing the exported sklearn model.

## Importing models for quadra evaluation

Quadra exported models are fully compatible with quadra evaluation tasks, this is possible because quadra uses a model wrapper emulating the standard pytorch interface for all the exported models. 

### Standard inference configuration

Evaluation models are regulated by a configuration file located at `configs/inference/default.yaml` shown below:

```yaml
onnx:
  session_options:
    inter_op_num_threads: 8
    intra_op_num_threads: 8
    graph_optimization_level:
      _target_: onnxruntime.GraphOptimizationLevel
      value: 99 # ORT_ENABLE_ALL
    enable_mem_pattern: true
    enable_cpu_mem_arena: true
    enable_profiling: false
    enable_mem_reuse: true
    execution_mode:
      _target_: onnxruntime.ExecutionMode
      value: 0 # ORT_SEQUENTIAL
    execution_order:
      _target_: onnxruntime.ExecutionOrder
      value: 0 # DEFAULT
    log_severity_level: 2
    log_verbosity_level: 0
    logid: ""
    optimized_model_filepath: ""
    use_deterministic_compute: false
    profile_file_prefix: onnxruntime_profile_

pytorch:
torchscript:
```

Right now we support custom option only for ONNX runtime, but we plan to add more inference configuration options in the future.