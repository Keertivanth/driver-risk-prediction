import os
import joblib
import pandas as pd


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_driver_risk_model.pkl"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_accident(
    weather,
    road_type,
    time_of_day,
    traffic_density,
    speed_limit,
    number_of_vehicles,
    driver_alcohol,
    road_condition,
    vehicle_type,
    driver_age,
    driver_experience,
    road_light_condition
):

    input_data = pd.DataFrame({
        "Weather": [weather],
        "Road_Type": [road_type],
        "Time_of_Day": [time_of_day],
        "Traffic_Density": [traffic_density],
        "Speed_Limit": [speed_limit],
        "Number_of_Vehicles": [number_of_vehicles],
        "Driver_Alcohol": [driver_alcohol],
        "Road_Condition": [road_condition],
        "Vehicle_Type": [vehicle_type],
        "Driver_Age": [driver_age],
        "Driver_Experience": [driver_experience],
        "Road_Light_Condition": [road_light_condition]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    prediction, probability = predict_accident(
        weather="Rainy",
        road_type="Highway",
        time_of_day="Night",
        traffic_density=2,
        speed_limit=80,
        number_of_vehicles=5,
        driver_alcohol=1,
        road_condition="Wet",
        vehicle_type="Car",
        driver_age=25,
        driver_experience=4,
        road_light_condition="No Light"
    )

    print("Prediction:", prediction)

    print(
        "Accident Probability:",
        f"{probability * 100:.2f}%"
    )