from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import os
import json

# Setting a dictionary to hold range-based feature groupings
RANGE_FEAT_GROUPS = {
    "Time_spent_Alone": {
        "introvert": [5, 6, 7, 8, 9, 10, 11],
        "ambivert": [4],
        "extrovert": [0, 1, 2, 3]
    },
    "Social_event_attendance": {
        "introvert": [0, 1, 2],
        "ambivert": [3],
        "extrovert": [4, 5, 6, 7, 8, 9, 10]
    },
    "Going_outside": {
        "introvert": [0, 1, 2],
        "ambivert": [3],
        "extrovert": [4, 5, 6, 7]
    },
    "Friends_circle_size": {
        "introvert": [0, 1, 2, 3],
        "ambivert": [4, 5],
        "extrovert": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    },
    "Post_frequency": {
        "introvert": [0, 1, 2],
        "ambivert": [3],
        "extrovert": [4, 5, 6, 7, 8, 9, 10]
    }
}

# Setting a list of the binary features
BINARY_FEATS = [
    "Stage_fear",
    "Drained_after_socializing"
]

# Loading the range_feat_medians dictionary from the JSON file
if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    with open("../data/other/medians.json", "r") as f:
        range_feat_medians = json.load(f)
else:
    with open('medians.json', 'r') as f:
        range_feat_medians = json.load(f)

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, range_feat_groups=None, binary_feats=None, range_feat_medians=None):
        self.range_feat_groups = range_feat_groups
        self.binary_feats = binary_feats
        self.range_feat_medians = range_feat_medians

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.clone()

        for feat, groupings in self.range_feat_groups.items():
            group_col = f"{feat}_group"
            mapping_expr = pl.when(pl.col(feat).is_in(groupings["introvert"])).then(pl.lit("introvert")) \
                             .when(pl.col(feat).is_in(groupings["ambivert"])).then(pl.lit("ambivert")) \
                             .when(pl.col(feat).is_in(groupings["extrovert"])).then(pl.lit("extrovert")) \
                             .otherwise(None)
            df = df.with_columns(mapping_expr.alias(group_col))

        for feat in self.range_feat_groups:
            group_col = f"{feat}_group"
            other_group_cols = [f"{f}_group" for f in self.range_feat_groups if f != feat]
            most_common_group_expr = pl.concat_list(other_group_cols).list.eval(pl.element().mode()).list.get(0).fill_null("extrovert")
            median_map = self.range_feat_medians[feat]
            median_fill_expr = pl.when(most_common_group_expr == "introvert").then(median_map.get("introvert")) \
                                 .when(most_common_group_expr == "ambivert").then(median_map.get("ambivert")) \
                                 .otherwise(median_map.get("extrovert"))
            df = df.with_columns(
                pl.col(feat).fill_null(median_fill_expr).alias(feat),
                pl.col(group_col).fill_null(most_common_group_expr).alias(group_col)
            )

        all_group_cols = [f"{f}_group" for f in self.range_feat_groups]
        most_common_overall_group = pl.concat_list(all_group_cols).list.eval(pl.element().mode()).list.get(0).fill_null("extrovert")
        fill_value = pl.when(most_common_overall_group == "introvert").then(pl.lit("Yes")).otherwise(pl.lit("No"))

        for feat in self.binary_feats:
            other_feat = next(f for f in self.binary_feats if f != feat)
            df = df.with_columns(
                pl.when(pl.col(feat).is_null())
                .then(pl.col(other_feat).fill_null(fill_value))
                .otherwise(pl.col(feat))
                .alias(feat)
            )

        label_cols = [f"{feat}_group" for feat in self.range_feat_groups] + self.binary_feats
        df = df.to_dummies(columns=label_cols)

        return df.to_pandas()
