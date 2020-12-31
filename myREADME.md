## Setup

conda create --name dh-mkt-ds python=3.8

conda activate dh-mkt-ds

python -m ipykernel install --user --name=dh-mkt-ds

## Target

- Imbalanced dataset: ~22% positive
 using sample weights can improve results


## Features


## Classifier model


###Metric
 - Primary: f1 score (suitable for imbalanced data)
 - Plot confusion matrix
 - Other possibilites: ROC AUC, MCC (Matthews Correlation Coefficient)
 - Depending on the application of the model we could have our own custom metric (for instance F-beta with beta chosen according to business criteria)

###Algorithm choice
 
 In my work, I usually start off with a boosting algorithm like XGBoost, some of the main reasons for this are:

 - Good performance out of the box, even without hyperparameter tuning
 - Doesn't require scaling
 - it supports sample weights
 
For ease of implementation, instead of XGBoost library this time I'll use the sklearn's GradientBoostingClassifier

I did a quick comparison based on f1 performance on validation set of the following algorithms:
 - GradientBoostingClassifier (with and without sample weights)
 - RandomForestClassifier
 - LogisticRegression
 - SVC ( discarded without finishing training due to long running time. Convergence time could be improved by doing feature scaling).

GradientBoostingClassifier with sample weights showed a much better performance out-of-the-box (F1=0.584)


###Hyperaparameter tuning
 - I struggled for a while because tuned results were worse than out-of-the-box performance. Then I realized I wasn't passing sample weights to gridsearch.
 - Some simplifications I did for the sake of running time:
  - The chosen grid is very simplistic
  - I'm using only 3 folds. Usually I use 4 or 5 if there's enough data and time.
  - I'm using grid search, but in a real-life scenario I would prefer some bayesian optimizer like hyperopt. 

 - Hyperparameter grid

``` 
{
    "loss":["deviance"],
    "learning_rate": [0.01, 0.1],
    "min_samples_split": [200, 500],
    "min_samples_leaf": [20, 50],
    "max_depth":[3, 8],
    "max_features":["log2","sqrt"],
    "n_iter_no_change": [None, 10],
    "subsample":[0.5, 1.0],
    "n_estimators":[100, 200]
}
``` 
Note that we are including whether or not to use early stopping (`n_iter_no_change`)

 - Best hyperparameters:

	```
	{'learning_rate': 0.1,
	 'loss': 'deviance',
	 'max_depth': 8,
	 'max_features': 'sqrt',
	 'min_samples_leaf': 50,
	 'min_samples_split': 200,
	 'n_estimators': 100,
	 'subsample': 1.0}
	```

 - Best F1 score: 
 	Mean CV: 0.583
 	Re-fitted on all training data (scored on validation set): 0.5871

### Model evaluation

- Test set performance:
	- F1 score
	- ROC AUC curve
	- Confusion matrix

- Feature importances
	- Extract from fitted model
	- Given more time, I would use SHAP
