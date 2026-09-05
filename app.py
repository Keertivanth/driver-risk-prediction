import streamlit as st
import sys
import os


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

sys.path.append(SRC_DIR)


# ============================================================
# IMPORT PREDICTION FUNCTION
# ============================================================

from predict import predict_accident


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Driver Risk Prediction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🚗 Driver Risk Prediction System")

st.markdown(
    """
    ### AI-Based Accident Risk Prediction

    Enter the road, vehicle and driver information below
    to predict the probability of an accident.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About the Project")

st.sidebar.info(
    """
    This application uses a Machine Learning model
    to predict accident risk based on:

    • Weather conditions
    • Road type
    • Time of day
    • Traffic density
    • Speed limit
    • Number of vehicles
    • Driver alcohol status
    • Road condition
    • Vehicle type
    • Driver age
    • Driver experience
    • Road lighting
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Enter Driving Information")


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rainy",
            "Foggy",
            "Snowy",
            "Stormy"
        ]
    )


with col2:

    road_type = st.selectbox(
        "Road Type",
        [
            "City Road",
            "Rural Road",
            "Highway",
            "Mountain Road"
        ]
    )


with col3:

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    traffic_density = st.slider(
        "Traffic Density",
        min_value=0,
        max_value=3,
        value=1,
        step=1
    )


with col2:

    speed_limit = st.number_input(
        "Speed Limit",
        min_value=20,
        max_value=250,
        value=60,
        step=5
    )


with col3:

    number_of_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    driver_alcohol = st.selectbox(
        "Driver Alcohol",
        options=[
            0,
            1
        ],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    road_condition = st.selectbox(
        "Road Condition",
        [
            "Dry",
            "Wet",
            "Icy",
            "Under Construction"
        ]
    )


with col3:

    vehicle_type = st.selectbox(
        "Vehicle Type",
        [
            "Car",
            "Truck",
            "Bus",
            "Motorcycle"
        ]
    )


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    driver_age = st.number_input(
        "Driver Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with col2:

    driver_experience = st.number_input(
        "Driver Experience (Years)",
        min_value=0,
        max_value=80,
        value=5,
        step=1
    )


with col3:

    road_light_condition = st.selectbox(
        "Road Light Condition",
        [
            "Daylight",
            "Artificial Light",
            "No Light"
        ]
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Accident Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        prediction, probability = predict_accident(
            weather=weather,
            road_type=road_type,
            time_of_day=time_of_day,
            traffic_density=traffic_density,
            speed_limit=speed_limit,
            number_of_vehicles=number_of_vehicles,
            driver_alcohol=driver_alcohol,
            road_condition=road_condition,
            vehicle_type=vehicle_type,
            driver_age=driver_age,
            driver_experience=driver_experience,
            road_light_condition=road_light_condition
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("Prediction Result")


        probability_percentage = (
            probability * 100
        )


        if prediction == 1:

            st.error(
                "⚠️ HIGH ACCIDENT RISK"
            )

            st.metric(
                "Accident Probability",
                f"{probability_percentage:.2f}%"
            )

            st.warning(
                "The model predicts an increased "
                "likelihood of an accident."
            )

        else:

            st.success(
                "✅ LOW ACCIDENT RISK"
            )

            st.metric(
                "Accident Probability",
                f"{probability_percentage:.2f}%"
            )

            st.info(
                "The model predicts a lower "
                "likelihood of an accident."
            )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.write("Risk Probability")

        st.progress(
            min(
                max(
                    probability_percentage / 100,
                    0.0
                ),
                1.0
            )
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Driver Risk Prediction System | "
    "Machine Learning Project"
)