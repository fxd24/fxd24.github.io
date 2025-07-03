---
title: MLFlow CheatSheet
date: 2025-06-28
description: A quick reference guide for using MLFlow to manage the machine learning lifecycle, including tracking, logging, and model management.
tags: [MLFlow, ML, CheatSheet]
draft: false
---

 This cheat sheet provides a quick reference for using MLFlow, a popular open-source platform for managing the machine learning lifecycle.

## Tracking

### Basic Tracking

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("param1", 1)
    mlflow.log_metric("metric1", 0.87)
```

### Tracking with a specific run name

```python
import mlflow

with mlflow.start_run(run_name="My_Awesome_Run"):
    mlflow.log_param("param1", 1)
    mlflow.log_metric("metric1", 0.87)
```

### Tracking with a specific experiment

```python
import mlflow

mlflow.set_experiment("My_New_Experiment")

with mlflow.start_run():
    mlflow.log_param("param1", 1)
    mlflow.log_metric("metric1", 0.87)
```

### Logging multiple metrics/params

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"param1": 1, "param2": "value"})
    mlflow.log_metrics({"metric1": 0.87, "metric2": 0.13})
```

### Logging artifacts (files)

```python
import mlflow
import os

# Create a dummy file
with open("my_artifact.txt", "w") as f:
    f.write("Hello, MLFlow!")

with mlflow.start_run():
    mlflow.log_artifact("my_artifact.txt")
    # You can also log to a specific artifact path within the run
    mlflow.log_artifact("my_artifact.txt", artifact_path="data")

os.remove("my_artifact.txt")
```

