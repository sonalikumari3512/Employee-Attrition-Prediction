import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction System")

st.write(
    "Predict whether an employee is likely to leave the company "
    "using a Machine Learning model."
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "attrition_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
TRAIN_PATH = BASE_DIR / "data" / "X_train.csv"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

df = pd.read_csv(DATA_PATH)
X_train = pd.read_csv(TRAIN_PATH)


# --------------------------------------------------
# Sidebar - Employee Details
# --------------------------------------------------

st.sidebar.header("Employee Details")


age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

daily_rate = st.sidebar.number_input(
    "Daily Rate",
    min_value=100,
    max_value=1500,
    value=800
)

distance = st.sidebar.slider(
    "Distance From Home",
    1,
    30,
    5
)

education = st.sidebar.selectbox(
    "Education",
    [1, 2, 3, 4, 5]
)

employee_number = st.sidebar.number_input(
    "Employee Number",
    min_value=1,
    max_value=2000,
    value=1000
)

environment_satisfaction = st.sidebar.selectbox(
    "Environment Satisfaction",
    [1, 2, 3, 4]
)

hourly_rate = st.sidebar.number_input(
    "Hourly Rate",
    min_value=30,
    max_value=100,
    value=60
)

job_involvement = st.sidebar.selectbox(
    "Job Involvement",
    [1, 2, 3, 4]
)

job_level = st.sidebar.selectbox(
    "Job Level",
    sorted(df["JobLevel"].unique())
)

job_satisfaction = st.sidebar.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=50000,
    value=10000
)

monthly_rate = st.sidebar.number_input(
    "Monthly Rate",
    min_value=2000,
    max_value=30000,
    value=15000
)

num_companies_worked = st.sidebar.slider(
    "Number of Companies Worked",
    0,
    10,
    2
)

percent_salary_hike = st.sidebar.slider(
    "Percent Salary Hike",
    10,
    30,
    15
)

performance_rating = st.sidebar.selectbox(
    "Performance Rating",
    [1, 2, 3, 4]
)

relationship_satisfaction = st.sidebar.selectbox(
    "Relationship Satisfaction",
    [1, 2, 3, 4]
)

stock_option_level = st.sidebar.selectbox(
    "Stock Option Level",
    [0, 1, 2, 3]
)

total_working_years = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    8
)

training_times_last_year = st.sidebar.slider(
    "Training Times Last Year",
    0,
    10,
    3
)

work_life_balance = st.sidebar.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4]
)

years_company = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    5
)

years_current_role = st.sidebar.slider(
    "Years In Current Role",
    0,
    20,
    3
)

years_last_promotion = st.sidebar.slider(
    "Years Since Last Promotion",
    0,
    15,
    2
)

years_manager = st.sidebar.slider(
    "Years With Current Manager",
    0,
    20,
    3
)


# --------------------------------------------------
# Categorical Inputs
# --------------------------------------------------

business_travel = st.sidebar.selectbox(
    "Business Travel",
    sorted(df["BusinessTravel"].unique())
)

department = st.sidebar.selectbox(
    "Department",
    sorted(df["Department"].unique())
)

education_field = st.sidebar.selectbox(
    "Education Field",
    sorted(df["EducationField"].unique())
)

gender = st.sidebar.selectbox(
    "Gender",
    sorted(df["Gender"].unique())
)

job_role = st.sidebar.selectbox(
    "Job Role",
    sorted(df["JobRole"].unique())
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    sorted(df["MaritalStatus"].unique())
)

overtime = st.sidebar.selectbox(
    "OverTime",
    ["No", "Yes"]
)


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "DailyRate": [daily_rate],
    "DistanceFromHome": [distance],
    "Education": [education],
    "EmployeeNumber": [employee_number],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobSatisfaction": [job_satisfaction],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked],
    "PercentSalaryHike": [percent_salary_hike],
    "PerformanceRating": [performance_rating],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "StockOptionLevel": [stock_option_level],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training_times_last_year],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_company],
    "YearsInCurrentRole": [years_current_role],
    "YearsSinceLastPromotion": [years_last_promotion],
    "YearsWithCurrManager": [years_manager],
    "BusinessTravel": [business_travel],
    "Department": [department],
    "EducationField": [education_field],
    "Gender": [gender],
    "JobRole": [job_role],
    "MaritalStatus": [marital_status],
    "OverTime": [overtime]
})


# --------------------------------------------------
# Display Employee Information
# --------------------------------------------------

st.subheader("Employee Information")

st.dataframe(input_data)


# --------------------------------------------------
# Encode Categorical Variables
# --------------------------------------------------

input_encoded = pd.get_dummies(input_data)


# --------------------------------------------------
# Match Training Features Exactly
# --------------------------------------------------

training_columns = X_train.columns

input_encoded = input_encoded.reindex(
    columns=training_columns,
    fill_value=0
)


# --------------------------------------------------
# Scale Numeric Features
# --------------------------------------------------

numeric_columns = list(scaler.feature_names_in_)

input_encoded[numeric_columns] = scaler.transform(
    input_encoded[numeric_columns]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Attrition"):

    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "⚠️ Employee is likely to Leave the Company."
        )
    else:
        st.success(
            "✅ Employee is likely to Stay in the Company."
        )

    st.metric(
        "Probability of Leaving",
        f"{probability * 100:.2f}%"
    )


# --------------------------------------------------
# Information
# --------------------------------------------------

st.info(
    """
    Prediction is based on the trained Logistic Regression
    classification model.

    Higher probability indicates a greater likelihood of
    employee attrition.
    """
)

st.markdown("---")

st.caption(
    "Built using Python, Scikit-learn and Streamlit • Sonali Kumari"
)
