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



