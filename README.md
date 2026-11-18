# Student Performance Prediction Using Machine Learning

## Project Description
This project predicts students' final exam marks based on their study hours, attendance, previous marks, assignment scores, and sleep hours using Machine Learning.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Machine Learning Algorithm
Random Forest Regressor

## Input Features
- Study Hours
- Attendance
- Previous Marks
- Assignment Score
- Sleep Hours

## Model Evaluation Results
- Mean Absolute Error (MAE): 3.54
- Mean Squared Error (MSE): 20.43
- R² Score: 0.93

Note: These results are based on a small sample dataset.

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run main.py
```

## Project Files
- `main.py` – Streamlit web application
- `model.py` – Machine learning model and evaluation
- `student_data.csv` – Student dataset
- `requirements.txt` – Required Python libraries
- `.gitignore` – Files excluded from Git
- `README.md` – Project documentation