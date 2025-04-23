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

SPLIT_RATIOS = [0.6, 0.3, 0.1]


# data generation 

def gen_price(house) -> int:
    _base_price = int(house["SQUARE_FEET"] * 150)
    _price = int(
        _base_price
        + (10000 * house["NUM_BEDROOMS"])
        + (15000 * house["NUM_BATHROOMS"])
        + (15000 * house["LOT_ACRES"])
        + (15000 * house["GARAGE_SPACES"])
        - (5000 * (MAX_YEAR - house["YEAR_BUILT"]))
    )
    return _price


def gen_houses(num_houses) -> pd.DataFrame:
    _house_list = []
    for _ in range(num_houses):
        _house = {
            "SQUARE_FEET": int(np.random.normal(3000, 750)),
            "NUM_BEDROOMS": np.random.randint(2, 7),
            "NUM_BATHROOMS": np.random.randint(2, 7) / 2,
            "LOT_ACRES": round(np.random.normal(1.0, 0.25), 2),
            "GARAGE_SPACES": np.random.randint(0, 4),
            "YEAR_BUILT": min(MAX_YEAR, int(np.random.normal(1995, 10))),
        }
        _price = gen_price(_house)
        # column names/features
        _house_list.append(
            [
                _price,
                _house["YEAR_BUILT"],
                _house["SQUARE_FEET"],
                _house["NUM_BEDROOMS"],
                _house["NUM_BATHROOMS"],
                _house["LOT_ACRES"],
                _house["GARAGE_SPACES"],
            ]
        )
    # convert the list to a DataFrame
    _df = pd.DataFrame(
        _house_list,
        columns=COLUMNS,
    )
    return _df


def split_data(df: pd.DataFrame, seed: int, 
               split: typing.List[float]
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pass

