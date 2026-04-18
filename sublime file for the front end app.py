import streamlit as st
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Updated containers (removed beta_)
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


# Cache updated (new syntax)
@st.cache_data
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data


with header:
    st.title("Welcome to this page ANSHU TRIPATHI")
    st.text("In this project, I analyze NYC taxi trip data.")


with dataset:
    st.header('NYC Taxi Dataset')
    st.text("Dataset source: NYC Taxi data")

    taxi_data = get_data("data/NYC taxi.csv")
    st.write(taxi_data.head())
    
    st.subheader("Pick-up Location ID Distribution")
    pulocation_dist = pd.DataFrame(taxi_data["PULocationID"].value_counts())
    st.bar_chart(pulocation_dist)


with features:
    st.header("Features Used")

    st.markdown("* **Feature 1:** PULocationID used to understand pickup trends")
    st.markdown("* **Feature 2:** Trip distance used as target variable")


with modelTraining:
    st.header("Model Training")

    st.text("Adjust hyperparameters and see model performance")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider(
        "Max depth of model?", min_value=10, max_value=100, value=20, step=10
    )

    n_estimators = sel_col.selectbox(
        "Number of trees?", options=[100, 200, 300, 400, 500, 600], index=0
    )

    sel_col.text("Available features:")
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input(
        "Input feature:", 'PULocationID'
    )

    # Model
    regr = RandomForestRegressor(
        max_depth=max_depth, n_estimators=n_estimators
    )

    x = taxi_data[[input_feature]]
    y = taxi_data["trip_distance"]

    regr.fit(x, y)
    prediction = regr.predict(x)

    disp_col.subheader("Mean Absolute Error")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean Squared Error")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R² Score")
    disp_col.write(r2_score(y, prediction))


if st.checkbox("Like it or Love it"):
    st.text("Thank you!")

st.text("Project inspired by my personal interest")
