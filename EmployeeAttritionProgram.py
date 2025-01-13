import streamlit as st #for web app
import joblib #for loading model
import pandas as pd #to manipulate inputted data
from sklearn import preprocessing #for label encoding

#1) Load model. 'rb' to read binary values
with open("employeeAttritionModel.pkl", "rb") as file:
  model = joblib.load(file)
print(type(model))

#title of web app
st.title("Employee Attrition Prediction")

#instructions for user
st.write("Enter employee details to predict likelihood of attrition.")

#2) Input fields
age = st.number_input("Age", min_value=18, max_value=100)
monthly_income = st.number_input("Monthly Income")
overtime = st.selectbox("Work Overtime?", ["Yes", "No"]) #Yes=1
total_working_years = st.number_input("Total Working Years")
business_travel = st.selectbox("Travel for Business?", ["Travel Rarely", "Travel Frequently", "No Travel"])
years_at_company = st.slider("Years at company", 0, 30, 5)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
hourly_rate = st.number_input("Hourly Rate")
daily_rate = st.number_input("Daily Rate")

#3) Prepare input data as df to pass it onto model
input_data = pd.DataFrame({
  "Age": [age], #input field under "column name"
  "MonthlyIncome": [monthly_income],
  "OverTime": [overtime],
  "TotalWorkingYears": [total_working_years],
  "BusinessTravel": [business_travel],
  "YearsAtCompany": [years_at_company],
  "MaritalStatus": [marital_status],
  "HourlyRate": [hourly_rate],
  "DailyRate": [daily_rate],
})



#4) Transform data
label_encoder = preprocessing.LabelEncoder()
#label-encoding
input_data["OverTime"] = label_encoder.fit_transform(input_data["OverTime"])

#onehot encoding
#manually force create one-hot column if BusinessTravel or MaritalStatus is represented
input_data["BusinessTravel_Travel_Frequently"] = input_data["BusinessTravel"].apply(lambda x:1 if x == "Travel Frequently" else 0)
input_data["BusinessTravel_Travel_Rarely"] = input_data["BusinessTravel"].apply(lambda x:1 if x == "Travel Rarely" else 0) 
input_data["MaritalStatus_Married"] = input_data["MaritalStatus"].apply(lambda x:1 if x == "Married" else 0) 
input_data["MaritalStatus_Single"] = input_data["MaritalStatus"].apply(lambda x:1 if x == "Single" else 0) 

#Drop original columns
input_data = input_data.drop(columns=["BusinessTravel", "MaritalStatus"])



#5) Clean input data
#conv all float columns to int
input_data["MonthlyIncome"] = input_data["MonthlyIncome"].astype(int)
input_data["TotalWorkingYears"] = input_data["TotalWorkingYears"].astype(int)
input_data["HourlyRate"] = input_data["HourlyRate"].astype(int)
input_data["DailyRate"] = input_data["DailyRate"].astype(int)

input_data.info()

#Check for missing columns and add them back in
expected_columns = model.feature_names_in_ #columns model expects
missing_columns = set(expected_columns) - set(input_data.columns) #identify difference between expected column and input data columns
for col in missing_columns: #adds missing columns into "input_data" with value 0
  input_data[col] = 0

#Reorder columns to match reference data
input_data = input_data[expected_columns]


#6) Make prediction using model
prediction = model.predict(input_data)

#7) Display result
if prediction == 1:
  st.write("The employee is likely to **<span style='color:red'>LEAVE</span>** the company", unsafe_allow_html=True)
else:
  st.write("The employee is likely to **<span style='color:green'>STAY</span>** at the company", unsafe_allow_html=True)