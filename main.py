
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Prediction")
st.write("Predict student final exam marks using Machine Learning.")

# Load dataset
data = pd.read_csv("student_data.csv")

# Handle missing values
data = data.fillna(data.mean(numeric_only=True))

# Separate input and output
X = data.drop("final_marks", axis=1)
y = data["final_marks"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Student input form
st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

previous_marks = st.number_input(
    "Previous Exam Marks",
    min_value=0,
    max_value=100,
    value=70
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0,
    max_value=100,
    value=65
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

# Prediction button
if st.button("Predict Student Performance"):

    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_marks,
          assignment_score, sleep_hours]],
        columns=X.columns
    )

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success("Prediction Completed!")

    st.metric(
        "Predicted Final Marks",
        f"{prediction:.2f} / 100"
    )

    if prediction >= 75:
        st.write("Performance Category: Excellent")
    elif prediction >= 50:
        st.write("Performance Category: Average")
    else:
        st.write("Performance Category: Needs Improvement")

    st.progress(int(prediction))
