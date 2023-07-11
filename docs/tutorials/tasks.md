# Tasks

Tasks are the main classes to interact with the experiments running using the `quadra` library. They implement functions as a pipeline of operations to be executed in a specific order. Following sections are explaining already implemented task pipelines.

## Task Flow

Each task has a set of functions to design the flow of each experiment. The `execute` function calls the given function below in a sequential way.

``` mermaid
graph LR;
    P(Prepare)-->TR(Train);
    TR-->TE(Test);
    TE-.->EM(Export model);
    EM-.->GR(Generate Report);
    GR-->F(Finalize);
```

!!! note
    You can extend or modify `execute` function or add new functions to change or modify the order of functions calling.

!!! note
    The `metadata` field of the task is used to store the results of each step in the flow. This field is suitable for using output of previous steps as input of the next step.

Tasks are also handling class instantiation and configuration from hydra configuration files.


## Class Inheritance
``` mermaid
graph LR;
    T(Task)-->L(LightningTask);
    T-->E(Evaluation);
    L-->SG(Segmentation);
    L-->CL(Classification);
    T-->SCL(SklearnClassification);
    T-->PSCL(PatchSklearnClassification);
    L-->SSL(SSL);
    L-->AD(AnomalibDetection)
```

- **[`Task`][quadra.tasks.base.LightningTask]:** Instantiate the DataModule, pretty much everything else must be implemented in the child classes.
- **[`Lightning Task`][quadra.tasks.base.LightningTask]:** Adds trainer, callbacks, logger and GPU parsing to the base task class as default. During the train and test phases it will use the `pytorch-lightning` library to train/test the model.
- **[`Evaluation Task`][quadra.tasks.base.Evaluation]:** Base task to load a model in the deployment format and run inference with it on new data, should be re-implemented for each task.
- **[`Segmentation Task`][quadra.tasks.Segmentation]:** It has the same functionality as the lightning task but it will also generate segmentation reports on demand.
- **[`Classification Task`][quadra.tasks.Classification]:** This task is designed to train from scratch or finetune a classification model using the `pytorch-lightning` library.
- **[`SklearnClassification Task`][quadra.tasks.classification.SklearnClassification]:** This task is designed to train an `sklearn` classifier on top of a torch feature extractor.
- **['PatchSklearnClassification'][quadra.tasks.classification.PatchSklearnClassification]:** This task is designed to train an `sklearn` patch classifier on top of a torch feature extractor.
- **[`Anomalib Detection Task`][quadra.tasks.AnomalibDetection]:** This task is designed to train an anomaly detection model using the `anomalib` library.
- **[`SSL (Self Supervised Learning) Task`][quadra.tasks.SSL]:** This task is designed to train a torch module with a given SSL algorithm.

Most of these tasks have an associated evaluation task used for inference.
## Adding new tasks

If your require the `pytorch-lightning` library, you can add a new task by extending the [`LightningTask`][quadra.base.LightningTask] class. Otherwise, you can simply start implementing a new task by extending the `Task` class.