import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import joblib
import pandas as pd
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATASET_COLUMNS = OrderedDict(
    {
        "#preg": int,
        "pgc_2h": int,
        "diastolic_bp": int,
        "tricep_skin_fold_mm": int,
        "serum_insulin_2h": int,
        "bmi": float,
        "diabetes_pedigree": float,
        "age": int,
        "class": int,
    }
)


FEATURE_COLUMNS = OrderedDict({k: v for k, v in DATASET_COLUMNS.items() if k != "class"})

CLASSES_COLUMNS = OrderedDict({"class": int})


@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def split_traintest_dataset(dataset: FlyteFile[typing.TypeVar("csv")], seed: int,
                            test_split_ratio: float
                            ) -> Tuple[FlyteSchema[FEATURE_COLUMNS],
                                    FlyteSchema[FEATURE_COLUMNS],
                                    FlyteSchema[CLASSES_COLUMNS],
                                    FlyteSchema[CLASSES_COLUMNS]]:
    

    column_names = [k for k in DATASET_COLUMNS.keys()]
    df = pd.read_csv(dataset, names=column_names)
    x = df[column_names[:8]]
    y = df[[column_names[-1]]]
    
    return train_test_split(x, y, test_size=test_split_ratio, random_state=seed)

MODELSER_JOBLIB = typing.TypeVar("joblib.dat")

@dataclass_json
@dataclass
class XGBoostModelHyperparams(object):
    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    booster: str = "gbtree"
    n_jobs: int = 1
    
model_file = typing.NamedTuple("Model", model=FlyteFile[MODELSER_JOBLIB])


@task
