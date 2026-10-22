# FSA MLOps

[![Train ML Model](https://github.com/gperdrizet/fsa-mlops/actions/workflows/train-model.yml/badge.svg)](https://github.com/gperdrizet/fsa-mlops/actions/workflows/train-model.yml)

## Quickstart

1. Fork the repository and create a Codespace on your fork (or clone locally)
2. Create a virtual environment and install the dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r ./requirements
```
## Demos

### 1. GitHub actions (CI/CD)

The script [`github-actions-demo/model.py`](https://github.com/gperdrizet/fsa-mlops/blob/main/github-actions-demo/model.py) trains a simple Scikit-learn `RandomForestClassifier()` model on the iris data. You can run it manually with:

```
python github-actions/model.py
```

This will train the model and save it as `iris_model.pkl`. You *could* then commit and push the trained model to GitHub for access by other people/applications (there are better places to store models!).

But, GitHub actions allows us to automate the environment set-up, retraining and upload of the model. The workflow file [train-model.yml](https://github.com/gperdrizet/fsa-mlops/blob/main/.github/workflows/train-model.yml).

Make a change to the model code, for example, increase the number of estimators in the ensemble. Then commit and push the code change to GitHub. 

```
git add github-actions/model.py
git commit -m "Increased number of estimators"
git push
```

The workflow triggers and the model is retrained on GitHub's servers. The resulting model file is attached to the specific workflow run, and the commit that triggered it. See the 'Actions' tab on the repository home page.

Actions let you automate practically anything and can be triggered by many different events: push, pull-request open/close/merge, tag, release - basically anything that GitHub tracks.

## 2. MLFlow

The script [`mlflow-demo/main.py`](https://github.com/gperdrizet/fsa-mlops/blob/main/demo-3/main.py) again trains a simple model on the iris data - but it uses MLFlow to track the training run, recording hyperparameters, performance results and retaining a copy of the trained model.

MLFlow also provides a web dashboard to view experiments. To start the MLFlow dashboard:

```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

Then visit port 5000 on localhost.