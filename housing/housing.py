import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def load_housing_data():

    csv_path = os.path.dirname(os.path.realpath(
        __file__)) + "/california_housing/housing.csv"
    return pd.read_csv(csv_path)


housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# housing_with_id = housing
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
# print(len(train_set), "train +", len(test_set), "test")

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:,
                                     population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)
# exit()
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# print(tree_rmse_scores)

# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# forest_predictions = forest_reg.predict(housing_prepared)
# forest_rmse = np.sqrt(mean_squared_error(housing_labels, forest_predictions))
# print(forest_rmse)
# print(np.sqrt(-cross_val_score(forest_reg, housing_prepared,
#       housing_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)))

svm = SVR(kernel="poly", degree=3, C=100)
svm.fit(housing_prepared, housing_labels)
housing_predictions = svm.predict(housing_prepared)
housing_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print(housing_rmse)
print(np.sqrt(-cross_val_score(svm, housing_prepared,
                               housing_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)))
