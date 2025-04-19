from flytekit import Resources, dynamic, workflow, task
from typing import Tuple
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from flytekit.types.file import JoblibSerializedFile
