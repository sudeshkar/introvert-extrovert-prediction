from Feature_engineering import FeatureEngineering
import polars as pl
import joblib
def load_model():
    return joblib.load("../models/personality_prediction_model2.pkl")
pipeline = load_model()
test_input = {
    "Time_spent_Alone": 5,
    "Stage_fear": "Yes",
    "Social_event_attendance": 5,
    "Going_outside": 10,
    "Drained_after_socializing": "No",
    "Friends_circle_size": 5,
    "Post_frequency": 5
}
input_df = pl.DataFrame([test_input])
prediction = pipeline.predict(input_df)
print(prediction)