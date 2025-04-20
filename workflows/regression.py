from flytekit import Resources, dynamic, workflow, task
from typing import Tuple
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from flytekit.types.file import JoblibSerializedFile



NUM_HOUSES_PER_LOCATION = 1000
COLUMNS = [
    "PRICE", "YEAR_BUILT", "SQUARE_FEET",
    "NUM_BEDROOMS", "NUM_BATHROOMS",
    "LOT_ACRES", "GARAGE_SPACES"
]
MAX_YEAR = 2021

