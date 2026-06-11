# MLflow Code Repository

A hands-on series of notebooks covering experiment tracking, hyperparameter tuning, model registration, and deployment using **MLflow** and **Optuna**.

---

## Setup

Navigate to the project folder and create the conda environment:

```cmd
conda env create -f mlflow_env.yaml
conda activate mlflow
```

Start the MLflow tracking server:

```cmd
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 8000
```

Then open `http://localhost:8000` in your browser to view the MLflow UI.

---

## Notebooks

| Notebook | Description |
|---|---|
| `MLFlow Demo.ipynb` | Introduction to MLflow — logging params, metrics, and artifacts |
| `MLFlow with Data Logging.ipynb` | Logging datasets and data profiles alongside model runs |
| `MLFlow Mentos Zindagi.ipynb` | End-to-end ML pipeline with MLflow tracking |
| `MLFlow Grid Search.ipynb` | Hyperparameter tuning with Grid Search + MLflow |
| `MLFlow Optuna.ipynb` | Advanced hyperparameter tuning with Optuna + MLflow |

---

## Optuna + MLflow (Featured)

`MLFlow Optuna.ipynb` is the most advanced notebook in this repo. It combines two powerful tools:

### Optuna
[Optuna](https://optuna.org/) is an automatic hyperparameter optimization framework. Unlike Grid Search or Random Search, Optuna uses the **TPE (Tree-structured Parzen Estimator)** algorithm — it learns from previous trials and focuses sampling on the most promising regions of the search space.

Key concepts used:
- `optuna.create_study(direction='maximize')` — defines the optimization goal
- `trial.suggest_int()`, `trial.suggest_categorical()` — define the search space
- `study.optimize(objective, n_trials=30)` — runs 30 intelligent trials
- `study.best_params`, `study.best_value` — retrieve the best result

### MLflow Integration
Every Optuna trial is tracked as a **nested MLflow child run** under a parent run:

```
Optuna_Study (parent run)
├── trial_0  (child run) — params + cv_accuracy logged
├── trial_1  (child run)
├── ...
└── trial_29 (child run)
```

The best model is then logged in a separate run and **registered in the MLflow Model Registry** for versioning and deployment.

### Workflow Summary

```
Load Data → Clean → Preprocessing Pipeline → Optuna Study (30 trials)
    → Best Params → Train Final Model → Log to MLflow → Register in Model Registry
```

### Loading the Registered Model

```python
import mlflow.pyfunc

mlflow.set_tracking_uri("http://localhost:8000")
model = mlflow.pyfunc.load_model("models:/RandomForest_Optuna_Titanic/1")
predictions = model.predict(X_test)
```

---

## Model Deployment

The registered model can be containerized and deployed using MLflow's built-in tooling:

```bash
# Build a Docker image with a REST API endpoint
mlflow models build-docker \
  --model-uri "models:/RandomForest_Optuna_Titanic/1" \
  --name "titanic-rf-model"

# Run locally and test
docker run -p 5001:8080 titanic-rf-model

From there the image can be pushed to **AWS ECR** and deployed on **ECS Fargate** or **SageMaker**.

---

## Common Gotchas

1. Not running the MLflow server before executing logging code in the notebook.
2. Logging parameters before model training completes — always log after fitting.
3. Not running all logging code in the same cell — partial runs leave incomplete records.
4. Writing parameter names manually in complex sklearn pipelines — let MLflow autolog handle it.
5. Forgetting `return mean_accuracy` in the Optuna objective function — Optuna cannot optimize without it.


