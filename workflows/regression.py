from flytekit import Resources, dynamic, workflow
from typing import Tuple
import pandas
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

