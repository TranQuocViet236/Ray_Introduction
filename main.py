import logging
import ray
import numpy as np
import mars.dataframe as md
from mars.learn.model_selection  import train_test_split
from mars.learn.datasets import make_classification
from xgboost_ray import RayDMatrix, RayParams, train, predict


logger = logging.getLogger(__name__)
logging.basicConfig(format=ray.ray_constants.LOGGER_FORMAT, level=logging.INFO)


def _load_data(n_sample: int,
               n_features: int,
               n_classes: int,
               test_size: float = 0.1,
               shuffle_seed: int = 42):

    n_informative = int(n_features*0.5)
    n_redundant = int(n_features*0.2)

    # generate data
    X, y  = make_classification(n_sample=n_sample, n_features=n_features,
                                n_classes=n_classes, n_informative=n_informative,
                                n_redundant=n_redundant, random_state = shuffle_seed)

    X, y = md.DataFrame(X), md.DataFrame({'label':y})

    X.columns = ['feature-' + str(i) for i in range(n_features)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=shuffle_seed)

    return md.concat([X_train, y_train],axis=1), md.concat([X_test, y_test], axis=1)

