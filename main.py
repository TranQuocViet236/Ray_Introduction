import logging
import ray
import numpy as np
import mars
import mars.dataframe as md
from mars.learn.model_selection import train_test_split
from mars.learn.datasets import make_classification
from xgboost_ray import RayDMatrix, RayParams, train, predict

logger = logging.getLogger(__name__)
logging.basicConfig(format=ray.ray_constants.LOGGER_FORMAT, level=logging.INFO)

def _load_data(n_samples: int,
               n_features:int,
               n_classes: int,
               test_size: float = 0.1,
               shuffle_seed: int = 42):
    n_informative = int(n_features * 0.5)
    n_redundant = int(n_features * 0.2)
    # generate dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, n_informative=n_informative,
                               n_redundant=n_redundant, random_state=shuffle_seed)
    X, y = md.DataFrame(X), md.DataFrame({"labels": y})
    X.columns = ['feature-' + str(i) for i in range(n_features)]
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=shuffle_seed)
    return md.concat([X_train, y_train], axis=1), md.concat([X_test, y_test], axis=1)

def main(*args):
    n_samples, n_features, worker_num, worker_cpu, num_shards = 10 ** 5, 20, 10, 8, 10
    ray_params = RayParams(
        num_actors=10,
        cpus_per_actor=8
    )

    # setup mars
    mars.new_ray_session(worker_num=worker_num, worker_cpu=worker_cpu, worker_mem=8 * 1024 ** 3)
    n_classes = 10
    df_train, df_test = _load_data(n_samples, n_features, n_classes, test_size=0.2)
    # convert mars DataFrame to Ray dataset
    ds_train = md.to_ray_dataset(df_train, num_shards=num_shards)
    ds_test = md.to_ray_dataset(df_test, num_shards=num_shards)
    train_set = RayDMatrix(data=ds_train, label="labels")
    test_set = RayDMatrix(data=ds_test, label="labels")

    evals_result = {}
    params = {
        'nthread': 1,
        'objective': 'multi:softmax',
        'eval_metric': ['mlogloss', 'merror'],
        'num_class': n_classes,
        'eta': 0.1,
        'seed': 42
    }
    bst = train(
        params=params,
        dtrain=train_set,
        num_boost_round=200,
        evals=[(train_set, 'train')],
        evals_result=evals_result,
        verbose_eval=100,
        ray_params=ray_params
    )
    # predict on a test set.
    pred = predict(bst, test_set, ray_params=ray_params)
    precision = (ds_test.dataframe['labels'].to_pandas() == pred).astype(int).sum() / ds_test.dataframe.shape[0]
    logger.info("Prediction Accuracy: %.4f", precision)

main()