import os
import json
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        # Loading the range_feat_medians dictionary from the JSON file
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            with open("../data/other/medians.json", "r") as f:
                range_feat_medians = json.load(f)
        else:
            with open('medians.json', 'r') as f:
                range_feat_medians = json.load(f)
            # Setting a list of the binary features
        BINARY_FEATS = [
            "Stage_fear",
            "Drained_after_socializing"
        ]
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
        self.range_feat_groups = RANGE_FEAT_GROUPS
        self.binary_feats = BINARY_FEATS
        self.range_feat_medians = range_feat_medians

    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        # Convert to Polars if it's a Pandas DataFrame
        original_was_pandas = isinstance(df, pd.DataFrame)
        if original_was_pandas:
            df = pl.from_pandas(df)
        
        # Validate input columns
        expected_cols = list(self.range_feat_groups.keys()) + self.binary_feats
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.clone()

        # ADDING GROUP LABELS FOR RANGE-BASED FEATURES
        for feat, groupings in self.range_feat_groups.items():
            group_col = f"{feat}_group"
            mapping_expr = (
                pl.when(pl.col(feat).is_in(groupings["introvert"])).then(pl.lit("introvert"))
                .when(pl.col(feat).is_in(groupings["ambivert"])).then(pl.lit("ambivert"))
                .when(pl.col(feat).is_in(groupings["extrovert"])).then(pl.lit("extrovert"))
                .otherwise(None)
            )
            df = df.with_columns(mapping_expr.alias(group_col))

        # FILLING NULLS IN RANGE-BASED FEATURES
        for feat in self.range_feat_groups:
            group_col = f"{feat}_group"
            other_group_cols = [f"{other_feat}_group" for other_feat in self.range_feat_groups if other_feat != feat]
            most_common_group_expr = (
                pl.concat_list(other_group_cols)
                .list.eval(pl.element().mode())
                .list.get(0)
                .fill_null("extrovert")
            )
            median_map = self.range_feat_medians[feat]
            median_fill_expr = (
                pl.when(most_common_group_expr == "introvert").then(median_map.get("introvert"))
                .when(most_common_group_expr == "ambivert").then(median_map.get("ambivert"))
                .otherwise(median_map.get("extrovert"))
            )
            df = df.with_columns(
                pl.col(feat).fill_null(median_fill_expr).cast(pl.Float64).alias(feat),
                pl.col(group_col).fill_null(most_common_group_expr).alias(group_col)
            )

        # FILLING NULLS IN BINARY FEATURES
        all_group_cols = [f"{f}_group" for f in self.range_feat_groups]
        most_common_overall_group = (
            pl.concat_list(all_group_cols)
            .list.eval(pl.element().mode())
            .list.get(0)
            .fill_null("extrovert")
        )
        fill_value = pl.when(most_common_overall_group == "introvert").then(pl.lit("Yes")).otherwise(pl.lit("No"))
        for feat in self.binary_feats:
            other_feat = next(f for f in self.binary_feats if f != feat)
            df = df.with_columns(
                pl.when(pl.col(feat).is_null())
                .then(pl.col(other_feat).fill_null(fill_value))
                .otherwise(pl.col(feat))
                .alias(feat)
            )

        # ONE-HOT ENCODING LABEL-BASED FEATURES
        group_label_cols = [f"{feat}_group" for feat in self.range_feat_groups]
        binary_label_cols = self.binary_feats
        label_cols = group_label_cols + binary_label_cols
        df = df.to_dummies(columns=label_cols)

        # Ensure all expected columns are present
        expected_features = [
            'Time_spent_Alone', 'Stage_fear_No', 'Stage_fear_Yes', 'Social_event_attendance', 'Going_outside',
            'Drained_after_socializing_No', 'Drained_after_socializing_Yes', 'Friends_circle_size', 'Post_frequency',
            'Time_spent_Alone_group_ambivert', 'Time_spent_Alone_group_extrovert', 'Time_spent_Alone_group_introvert',
            'Social_event_attendance_group_ambivert', 'Social_event_attendance_group_extrovert',
            'Social_event_attendance_group_introvert', 'Going_outside_group_ambivert', 'Going_outside_group_extrovert',
            'Going_outside_group_introvert', 'Friends_circle_size_group_ambivert', 'Friends_circle_size_group_extrovert',
            'Friends_circle_size_group_introvert', 'Post_frequency_group_ambivert', 'Post_frequency_group_extrovert',
            'Post_frequency_group_introvert'
        ]
        for col in expected_features:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).cast(pl.Int32).alias(col))
        df = df.select(expected_features)

        # Convert back to Pandas if the input was originally Pandas
        if original_was_pandas:
            return df.to_pandas()
        return df