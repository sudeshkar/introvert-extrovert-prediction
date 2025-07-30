from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import pandas as pd
import os
import json

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

BINARY_FEATS = ["Stage_fear", "Drained_after_socializing"]

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.range_feat_groups = RANGE_FEAT_GROUPS
        self.binary_feats = BINARY_FEATS
        self.feature_names_ = []

        try:
            with open("../data/other/medians.json", "r") as f:
                self.range_feat_medians = json.load(f)
        except FileNotFoundError:
            self.range_feat_medians = {
                "Time_spent_Alone": {"introvert": 7, "ambivert": 4, "extrovert": 2},
                "Social_event_attendance": {"introvert": 1, "ambivert": 3, "extrovert": 6},
                "Going_outside": {"introvert": 1, "ambivert": 3, "extrovert": 5},
                "Friends_circle_size": {"introvert": 2, "ambivert": 4, "extrovert": 8},
                "Post_frequency": {"introvert": 1, "ambivert": 3, "extrovert": 6}
            }

    def fit(self, X, y=None):
        df = self.transform(X)
        self.feature_names_ = df.columns.tolist()
        return self

    def transform(self, df):
        df = df.clone()

        for feat, groupings in self.range_feat_groups.items():
            if feat in df.columns:
                group_col = f"{feat}_group"
                mapping_expr = pl.when(pl.col(feat).is_in(groupings["introvert"])).then("introvert") \
                    .when(pl.col(feat).is_in(groupings["ambivert"])).then("ambivert") \
                    .when(pl.col(feat).is_in(groupings["extrovert"])).then("extrovert") \
                    .otherwise(None)
                df = df.with_columns(mapping_expr.alias(group_col))

        for feat in self.range_feat_groups:
            if feat in df.columns:
                group_col = f"{feat}_group"
                other_group_cols = [f"{other}_group" for other in self.range_feat_groups if other != feat and f"{other}_group" in df.columns]

                if other_group_cols:
                    most_common_group_expr = pl.concat_list(other_group_cols).list.eval(pl.element().mode()).list.get(0).fill_null("extrovert")
                else:
                    most_common_group_expr = pl.lit("extrovert")

                median_map = self.range_feat_medians[feat]
                median_fill_expr = pl.when(most_common_group_expr == "introvert").then(median_map["introvert"]) \
                    .when(most_common_group_expr == "ambivert").then(median_map["ambivert"]) \
                    .otherwise(median_map["extrovert"])

                df = df.with_columns(
                    pl.col(feat).fill_null(median_fill_expr).alias(feat),
                    pl.col(group_col).fill_null(most_common_group_expr).alias(group_col)
                )

        existing_binary_feats = [f for f in self.binary_feats if f in df.columns]

        if existing_binary_feats:
            all_group_cols = [f"{f}_group" for f in self.range_feat_groups if f"{f}_group" in df.columns]
            most_common_group_expr = pl.concat_list(all_group_cols).list.eval(pl.element().mode()).list.get(0).fill_null("extrovert") if all_group_cols else pl.lit("extrovert")
            fill_value = pl.when(most_common_group_expr == "introvert").then("Yes").otherwise("No")

            for feat in existing_binary_feats:
                other_feats = [f for f in existing_binary_feats if f != feat]
                if other_feats:
                    df = df.with_columns(
                        pl.when(pl.col(feat).is_null()).then(pl.col(other_feats[0]).fill_null(fill_value)).otherwise(pl.col(feat)).alias(feat)
                    )
                else:
                    df = df.with_columns(pl.col(feat).fill_null(fill_value).alias(feat))

        label_cols = [f"{f}_group" for f in self.range_feat_groups if f"{f}_group" in df.columns] + existing_binary_feats
        if label_cols:
            df = df.to_dummies(columns=label_cols)

        df_pd = df.to_pandas()

        if self.feature_names_:
            for col in self.feature_names_:
                if col not in df_pd.columns:
                    df_pd[col] = 0
            df_pd = df_pd[self.feature_names_]

        return df_pd
