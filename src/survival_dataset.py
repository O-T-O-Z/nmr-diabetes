from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class SurvivalDataset(BaseModel):
    """Dataset for survival data."""

    prefix: str
    X: Optional[pd.DataFrame] = None
    y_lower_bound: Optional[pd.Series] = None
    y_upper_bound: Optional[pd.Series] = None
    censored: Optional[pd.DataFrame] = None
    event: Optional[pd.DataFrame] = None
    X_train: Optional[pd.Series] = None
    y_lower_bound_train: Optional[pd.Series] = None
    y_upper_bound_train: Optional[pd.Series] = None
    censored_train: Optional[pd.DataFrame] = None
    X_val: Optional[pd.Series] = None
    y_lower_bound_val: Optional[pd.Series] = None
    y_upper_bound_val: Optional[pd.Series] = None
    censored_val: Optional[pd.DataFrame] = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """Initialize the SurvivalDataset class."""
        super().__init__(prefix="../datasets.nosync/", **kwargs)

    def load_data(self, filename: str) -> None:
        """
        Load the data from the given filename.

        :param filename: name of the file to load the data from.
        """
        df = pd.read_csv(self.prefix + filename)
        self.censored = df[["CENSORED"]]
        self.event = pd.DataFrame((self.censored == 0).values.astype(np.int32))

        df = df.drop(["CENSORED"], axis=1, inplace=False)
        self.X = df.drop(["upper_bound", "lower_bound"], axis=1, inplace=False)

        self.y_lower_bound = df[["lower_bound"]].reset_index(drop=True).iloc[:, 0]
        self.y_upper_bound = df[["upper_bound"]].reset_index(drop=True).iloc[:, 0]
        if filename.startswith("clinical") or filename.startswith("full"):
            self.X = self.X.fillna(self.X.median())

    def split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split the data into training and validation sets.

        :param test_size: portion of the validation split, defaults to 0.2
        :param random_state: shuffle seed, defaults to 42
        """
        (
            self.X_train,
            self.X_val,
            self.y_lower_bound_train,
            self.y_lower_bound_val,
            self.y_upper_bound_train,
            self.y_upper_bound_val,
        ) = train_test_split(
            self.X,
            self.y_lower_bound,
            self.y_upper_bound,
            test_size=test_size,
            random_state=random_state,
            stratify=self.censored,
        )
        self.censored_train = self.censored.loc[self.y_lower_bound_train.index]
        self.censored_val = self.censored.loc[self.y_lower_bound_val.index]

    def split_on_indices(self, indices: tuple[np.ndarray, np.ndarray | None]) -> None:
        """
        Split the data into training and validation sets based on the given indices.

        :param indices: tuple of training and validation indices. The validation indices can be None.
        """
        train_index, valid_index = indices

        self.X_train, self.y_lower_bound_train, self.y_upper_bound_train = (
            self.X.iloc[train_index],
            self.y_lower_bound.iloc[train_index],
            self.y_upper_bound.iloc[train_index],
        )
        self.censored_train = self.censored.loc[self.y_lower_bound_train.index]

        if valid_index is not None:
            self.X_val, self.y_lower_bound_val, self.y_upper_bound_val = (
                self.X.iloc[valid_index],
                self.y_lower_bound.iloc[valid_index],
                self.y_upper_bound.iloc[valid_index],
            )
            self.censored_val = self.censored.loc[self.y_lower_bound_val.index]
