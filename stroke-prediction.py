import streamlit as st

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.drop("id", inplace=True, axis=1)

data["bmi"].replace(to_replace=np.NaN, value=data["bmi"].mean(), inplace=True)

le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])
data["ever_married"] = le.fit_transform(data["ever_married"])
data["work_type"] = le.fit_transform(data["work_type"])
data["Residence_type"] = le.fit_transform(data["Residence_type"])
data["smoking_status"] = le.fit_transform(data["smoking_status"])

st.title("Stroke Prediction")

gender = st.radio("Pick your gender", ["Male", "Female", "Other"])

age = st.number_input("Enter your age", value=0, format="%d")


if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 0
else:
    gender = 2


hypertension = st.radio("Do you have hypertension?", ["Yes", "No"])

if hypertension == "Yes":
    hypertension = 1
else:
    hypertension = 0

heartDisease = st.radio("Do you have any heart disease?", ["Yes", "No"])

if heartDisease == "Yes":
    heartDisease = 1
else:
    heartDisease = 0

everMarried = st.radio("Have you ever been married?", ["Yes", "No"])

if everMarried == "Yes":
    everMarried = 1
else:
    everMarried = 0

workType = st.selectbox(
    "Choose your work type",
    ["Government Job", "Private", "Self-employed", "Children", "Never worked"],
)

if workType == "Government Job":
    workType = 0
elif workType == "Private":
    workType = 2
elif workType == "Self-employed":
    workType = 3
elif workType == "Never worked":
    workType = 1
else:
    workType = 4

residentalArea = st.selectbox("Choose your residental area", ["Rural", "Urban"])

if residentalArea == "Rural":
    residentalArea = 0
else:
    residentalArea = 1

glucoseLevel = st.number_input("Enter your average glucose level", format="%f")

weight = st.number_input("Enter your weight (in kg)", value=1.0)
height = st.number_input("Enter your height (in m)", value=1.0)
bmi = round(weight / (height ** 2), 1)
# st.write("The bmi is ", bmi)


smokingStatus = st.selectbox(
    "What is your smoking status?",
    ["Never smoked", "Smokes", "Formerly smoked", "Unknown"],
)


if smokingStatus == "Never smoked":
    smokingStatus = 2
elif smokingStatus == "Smokes":
    smokingStatus = 3
elif smokingStatus == "Formerly smoked":
    smokingStatus = 1
else:
    smokingStatus = 0


predict = st.button("Predict!")
newUser = [
    gender,
    age,
    hypertension,
    heartDisease,
    everMarried,
    workType,
    residentalArea,
    glucoseLevel,
    bmi,
    smokingStatus,
]

x = data.iloc[:, :-1].values  # bağımsız değişkenler
y = data.iloc[:, -1:].values  # bağımlı değişkenler

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100
)

sc = StandardScaler()
x_olcekli_train = sc.fit_transform(x_train)
x_olcekli_test = sc.transform(x_test)

y_olcekli_train = sc.fit_transform(y_train)
y_olcekli_test = sc.transform(y_train)

sm = SMOTE(random_state=2)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train.ravel())

newUser = pd.DataFrame(newUser)

classifier = XGBClassifier(eval_metric="error", learning_rate=0.1)
classifier.fit(x_train_sm, y_train_sm)

result = np.array(newUser).reshape((1, -1))

y_pred = classifier.predict(x_test)
y_pred_values = pd.DataFrame(y_pred).value_counts()

y_test_values = pd.DataFrame(y_test).value_counts()

percentageOfSick = (y_test_values[1] / (y_test_values[0] + y_test_values[1])) * 100
percentageOfHealthy = (y_test_values[0] / (y_test_values[0] + y_test_values[1])) * 100

pred1 = classifier.predict(np.array(newUser).reshape((1, -1)))  # 1
pred2 = classifier.predict(np.array(newUser).reshape((1, -1)))  # 0


if predict:
    if pred1 == 1:
        st.write(str(round((100 - percentageOfSick), 3)), "% chance you are sick.")
    if pred2 == 0:
        st.write(str(round(percentageOfHealthy, 3)), "% chance you are healthy.")
