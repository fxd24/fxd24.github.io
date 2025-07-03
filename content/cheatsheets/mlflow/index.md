---
title: "MLflow Cheat Sheet - Essential ML Experiment Tracking"
date: 2025-06-28
draft: false
tags: ["mlflow", "machine-learning", "experiment-tracking", "python", "cheat-sheet"]
categories: ["Data Science", "MLOps"]
summary: "Complete MLflow reference guide with essential commands, examples, and best practices for ML experiment tracking and model management."
---

## Why MLflow Matters

MLflow is the industry standard for ML experiment tracking, model versioning, and deployment. It solves critical problems in ML development:

- **Reproducibility**: Track every experiment with parameters, metrics, and artifacts
- **Collaboration**: Share experiments and models across teams
- **Model Management**: Version, stage, and deploy models systematically
- **Framework Agnostic**: Works with any ML library (scikit-learn, PyTorch, TensorFlow, etc.)

Perfect for data scientists who want to move from "notebook chaos" to systematic ML development.

## Quick Installation

```bash
pip install mlflow
# Optional: with extras for specific frameworks
pip install mlflow[extras] # sklearn, pytorch, tensorflow
```

## Essential API Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `mlflow.start_run()` | Start tracking a new run | `with mlflow.start_run():` |
| `mlflow.log_param()` | Log hyperparameters | `mlflow.log_param("lr", 0.01)` |
| `mlflow.log_metric()` | Log metrics | `mlflow.log_metric("accuracy", 0.95)` |
| `mlflow.log_artifact()` | Log files/models | `mlflow.log_artifact("model.pkl")` |
| `mlflow.log_model()` | Log ML models | `mlflow.sklearn.log_model(model, "model")` |
| `mlflow.set_experiment()` | Set/create experiment | `mlflow.set_experiment("my-experiment")` |
| `mlflow.set_tag()` | Add metadata tags | `mlflow.set_tag("model_type", "random_forest")` |
| `mlflow.log_artifacts()` | Log directory of files | `mlflow.log_artifacts("./plots")` |
| `mlflow.get_run()` | Retrieve run info | `run = mlflow.get_run(run_id)` |
| `mlflow.search_runs()` | Query experiments | `df = mlflow.search_runs(experiment_ids=["1"])` |

## Core Concepts with Examples

### Basic Experiment Tracking with Hugging Face Transformers

```python
import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np

# Set experiment
mlflow.set_experiment("text-classification")

# Hugging Face Transformers: State-of-the-art NLP models library
# Website: https://huggingface.co/transformers/
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Sample data preparation
texts = ["This is great!", "This is terrible!"]
labels = [1, 0]
dataset = Dataset.from_dict({
    "text": texts,
    "labels": labels
})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_labels", 2)
    mlflow.log_param("max_length", 512)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=10,
    )
    
    # Log training hyperparameters
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Log metrics
    mlflow.log_metric("final_loss", trainer.state.log_history[-1]["train_loss"])
    mlflow.log_metric("total_steps", trainer.state.global_step)
    
    # Log the model with tokenizer
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model"
    )
    
    # Log tags
    mlflow.set_tag("model_type", "transformer")
    mlflow.set_tag("task", "text_classification")
    mlflow.set_tag("framework", "transformers")
```

### Post-Run Logging with Run ID

```python
import mlflow
import numpy as np

# Start initial run
with mlflow.start_run() as run:
    # Log basic info
    mlflow.log_param("model", "neural_network")
    mlflow.log_metric("initial_loss", 0.8)
    
    # Store run ID for later use
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

# Later: Add more information to the same run
with mlflow.start_run(run_id=run_id):
    # Log additional metrics
    mlflow.log_metric("final_loss", 0.15)
    mlflow.log_metric("epochs", 100)
    
    # Log artifacts
    np.save("predictions.npy", [0.9, 0.8, 0.95])
    mlflow.log_artifact("predictions.npy")
    
    # Update tags
    mlflow.set_tag("status", "completed")
    mlflow.set_tag("performance", "good")

# Alternative: Get run info and log more data
run_info = mlflow.get_run(run_id)
print(f"Run status: {run_info.info.status}")
print(f"Parameters: {run_info.data.params}")
```

### MLflow in a Trainer Class

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MLflowTrainer:
    def __init__(self, model, experiment_name="default"):
        self.model = model
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("optimizer", "Adam")
            
            # Setup training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                val_loss, val_acc = self._validate(val_loader, criterion)
                
                # Log metrics for this epoch
                mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Log final model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Log model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            mlflow.log_param("total_parameters", total_params)
            mlflow.set_tag("framework", "pytorch")
            
            return self.model
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return val_loss/len(val_loader), 100.*correct/total

# Usage
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

trainer = MLflowTrainer(model, experiment_name="mnist-classification")
# trained_model = trainer.train(train_loader, val_loader, epochs=20)
```

### PyTorch Lightning Integration

PyTorch Lightning is a high-level framework that organizes PyTorch code to decouple research from engineering. It handles training loops, validation, and distributed training automatically.  
**Website:** https://lightning.ai/docs/pytorch/stable/

```python
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn as nn

class LightningModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Lightning automatically logs to MLflow
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Setup MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name="pytorch-lightning-experiment",
    tracking_uri="file:./mlruns"
)

# Train with automatic MLflow logging
model = LightningModel(lr=0.001)
trainer = pl.Trainer(
    max_epochs=10,
    logger=mlflow_logger,
    log_every_n_steps=50
)

# trainer.fit(model, train_loader, val_loader)

# The logger automatically logs:
# - Hyperparameters
# - Metrics from self.log()
# - Model checkpoints
# - System metrics
```

### Model Registration

```python
import mlflow
from mlflow.tracking import MlflowClient

# Method 1: Register during logging
with mlflow.start_run():
    # Train your model
    model = train_model()
    
    # Log and register model in one step
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="iris-classifier"
    )

# Method 2: Register existing model
client = MlflowClient()

# Get model URI from a previous run
model_uri = "runs:/{}/model".format(run_id)

# Register the model
model_version = mlflow.register_model(
    model_uri=model_uri,
    name="iris-classifier"
)

print(f"Model registered: {model_version.name}, version {model_version.version}")

# Method 3: Programmatic model management
# Create new registered model
client.create_registered_model(
    name="production-model",
    description="Production model for iris classification"
)

# Transition model to different stages
client.transition_model_version_stage(
    name="iris-classifier",
    version=1,
    stage="Production"
)

# List all registered models
for model in client.list_registered_models():
    print(f"Model: {model.name}")
    for version in model.latest_versions:
        print(f"  Version {version.version}: {version.current_stage}")
```

## MLflow UI Navigation

```bash
# Start MLflow UI
mlflow ui

# Custom port and host
mlflow ui --port 5001 --host 0.0.0.0

# Point to remote tracking server
mlflow ui --backend-store-uri postgresql://user:password@host:port/db
```

**UI Features:**
- **Experiments**: Compare runs, sort by metrics
- **Models**: View registered models and versions
- **Filtering**: Use search syntax like `metrics.accuracy > 0.9`
- **Run Comparison**: Select multiple runs to compare side-by-side

## Model Serving

```bash
# Serve model locally
mlflow models serve -m "models:/iris-classifier/Production" -p 1234

# Test the served model
curl -X POST -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}' \
  http://localhost:1234/invocations
```

## Framework-Specific Examples

### Scikit-learn Integration

Scikit-learn is the most popular machine learning library for Python, providing simple and efficient tools for data mining and analysis.  
**Website:** https://scikit-learn.org/

```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Auto-logging (captures params, metrics, model)
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # Everything logged automatically!
```

## Configuration & Best Practices

### Environment Setup

```python
# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")  # Local
mlflow.set_tracking_uri("http://mlflow-server:5000")  # Remote

# Environment variables
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "default"
```

### Cloud Platform Integration

**Azure ML Studio**: MLflow is integrated into Azure ML as the default experiment tracking solution.  
**Website:** https://docs.microsoft.com/en-us/azure/machine-learning/  

**Databricks**: Provides managed MLflow with automatic experiment tracking.  
**Website:** https://databricks.com/product/managed-mlflow  

**Authentication Issues**: Common pitfall when using cloud platforms - ensure your authentication tokens and tracking URIs are correctly configured. Many authentication failures occur due to:
- Expired tokens
- Incorrect service principal permissions
- Network/firewall restrictions
- Mismatched tracking server URLs

### Alternative: Weights & Biases

Weights & Biases (wandb) is another popular experiment tracking platform with advanced visualization capabilities.  
**Website:** https://wandb.ai/  

```python
# Quick comparison - wandb syntax
import wandb
wandb.init(project="my-project")
wandb.log({"accuracy": 0.95, "loss": 0.1})
```

### Batch Logging for Performance

```python
# Instead of logging metrics one by one
with mlflow.start_run():
    for epoch in range(100):
        # Don't do this - too many API calls
        mlflow.log_metric("loss", loss, step=epoch)
    
    # Better: Log in batches
    losses = []
    for epoch in range(100):
        losses.append(loss)
        if epoch % 10 == 0:  # Log every 10 epochs
            for i, loss_val in enumerate(losses):
                mlflow.log_metric("loss", loss_val, step=epoch-len(losses)+i+1)
            losses = []
```

### Organization Tips

```python
# Use nested runs for hyperparameter tuning
with mlflow.start_run(run_name="hyperparameter_tuning"):
    for lr in [0.01, 0.1, 0.001]:
        with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
            mlflow.log_param("learning_rate", lr)
            # Train and log results
```

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "No such file or directory: mlruns" | Run `mlflow ui` in directory containing mlruns/ |
| **Authentication failures (Azure/Databricks)** | **Verify tokens, service principal permissions, and tracking URIs** |
| Slow logging | Use batch logging, avoid logging in tight loops |
| Large artifacts | Use `log_artifact()` for files, not `log_param()` |
| Run not found | Check experiment ID and run ID are correct |
| Permission errors | Check file permissions on mlruns directory |
| **Cloud platform connection issues** | **Check network access, firewall rules, and endpoint URLs** |

## Quick Commands Reference

```bash
# CLI commands
mlflow ui                              # Start web UI
mlflow experiments list                # List experiments
mlflow runs list --experiment-id 1     # List runs
mlflow models serve -m model_uri       # Serve model
mlflow doctor                          # Check installation
```

## Advanced Features

### Advanced Features

### Custom Metrics and Plots

Matplotlib is the fundamental plotting library for Python, providing a MATLAB-like interface.  
**Website:** https://matplotlib.org/  

Seaborn is a statistical data visualization library based on matplotlib, providing high-level interface for attractive graphics.  
**Website:** https://seaborn.pydata.org/  

```python
import matplotlib.pyplot as plt
import mlflow

with mlflow.start_run():
    # Log custom plots
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig("loss_curves.png")
    mlflow.log_artifact("loss_curves.png")
    
    # Log confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
```

This cheat sheet covers the essential MLflow functionality you'll use daily. Keep it handy for quick reference during your ML development workflow!

**Additional Resources:**
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hugging Face Transformers Guide](https://huggingface.co/transformers/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Azure ML MLflow Integration](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow)
- [Databricks MLflow Guide](https://docs.databricks.com/mlflow/index.html)

---

*For more detailed information, visit the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).*