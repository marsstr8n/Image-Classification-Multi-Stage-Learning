# About Logistic Regression

- After my default LogisticRegression function gives the accuracy of 58%, I want to find best hyperparameters to improve this score - by using GridSearchCV

- First, I need to understand what are the arguments LogisticRegression take:
    - **penalty**: regularisation - how to prevent overfitting
        - 'l1': Lasso (can shrink some weights to zero → feature selection).
        - 'l2': Ridge (shrinks weights, but doesn't eliminate them).
        - 'elasticnet': Mix of L1 and L2 (only with saga solver).
        - 'none': No regularization (risk of overfitting).

    - **C**: inverse of regularisation strength
        - float numbers, e.g. 0.1, 1, 10
        - small C -> stronger regularisation
        - large C - > weaker regularisation (may overfit)

    - **solver**: optimisation algorithms to find the best weights (w values)
        - 'liblinear': Good for small datasets, supports L1 & L2.

        - 'lbfgs': Good for L2 and multiclass.

        - 'sag' / 'saga': Good for large datasets, supports all penalties (saga supports L1 too).

    - **max_iter**: number of iterations to converge

    - **multi_class**: strategy for multiclass classification
        - 'ovr' (one-vs-rest): Train one classifier per class.

        - 'multinomial': Treats all classes at once (better if softmax behavior is desired).

        - 'auto': Chooses based on solver.

    - **fit_intercept**: include a bias (intercept) in the model or not
        - default is 'True'

    - **class_weight**: balance weights of different classes
        - 'balanced': Automatically adjusts weights based on class frequencies.

        - {class_label: weight}: Manual adjustment.

    - **n_jobs**: how many CPU cores to use, only applies if solver supports parallelisation (e.g. liblinear)

## Logistic Regression combinations:
| Solver     | l1 | l2 | elasticnet | `l1_ratio` | Multiclass | Speed | Notes |
|------------|----|----|------------|------------|------------|-------|-------|
| `liblinear`| ✅  | ✅  | ❌         | ❌         | One-vs-Rest | ✅ Fast for small datasets | Only supports `'l1'` and `'l2'` |
| `lbfgs`    | ❌  | ✅  | ❌         | ❌         | Softmax (multi_class='multinomial') | ⚡ Fast | Does **not** support `'l1'` |
| `sag`      | ❌  | ✅  | ❌         | ❌         | Large datasets | ⚡ Very Fast | Only supports `'l2'` |
| `saga`     | ✅  | ✅  | ✅         | ✅         | Large datasets | ⚡⚡ Super Fast | Only solver that supports `elasticnet` |

- For this binary classification, here are the params
param_grid = [
    # liblinear supports only l1 and l2 penalties
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1],
        'max_iter': [1000]
    },
    # lbfgs and sag support only l2 penalty
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1],
        'max_iter': [1000]
    },
    # C = 1 not converge for 'sag'
    {
        'solver': ['sag'],
        'penalty': ['l2'],
        'C': [0.01, 0.1],
        'max_iter': [2000]
    },

    # saga with l1 or l2 (no l1_ratio needed). Note - C = [0.01, 1] long time to converge for 'saga' too
    {
        'solver': ['saga'],
        'penalty': ['l1', 'l2'],
        'C': [0.01],
        'max_iter': [1000]
    },
    # saga with elasticnet (l1_ratio required)
    {
        'solver': ['saga'],
        'penalty': ['elasticnet'],
        'C': [0.01],
        'l1_ratio': [0.3, 0.5, 0.7],
        'max_iter': [1000]
    }
]

best params, after using gridsearchcv: {'C': 0.01, 'l1_ratio': 0.3, 'max_iter': 1000, 'penalty': 'elasticnet', 'solver': 'saga'}, Accuracy: 0.6042794147123672

the result on the test data: **accuracy is 61.46%** - 4% improvement
