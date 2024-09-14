import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from diabetes_correlation_analysis import create_correlation_table
from diabetes_correlation_analysis import create_diabetes_pyramid
from diabetes_correlation_analysis import create_genhlth_bmi_diabetes
from diabetes_correlation_analysis import create_smoker_hvyalcoholconsump_diabetes
from diabetes_correlation_analysis import create_education_income_diabetes
from diabetes_correlation_analysis import create_highBP_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import simpledialog, messagebox
import tkinter as tk

# Load the CSV data
df = pd.read_csv("/Users/lauwaikwan/Documents/Data engineer/Diabetes_project_to_git/diabetes_balanced_patients.csv")

int_columns = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 
    'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 
    'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# Convert specified columns to int
df[int_columns] = df[int_columns].astype(int)

# Ensure 'BMI' remains as float
df['BMI'] = df['BMI'].astype(float)

#print(df.info())

"""Creating training model and tkinter """
training_data = pd.read_csv("/Users/lauwaikwan/Documents/Data engineer/Diabetes_project_to_git/diabetes_balanced_patients.csv")

training_data = training_data.rename(columns={'Diabetes_binary': 'Diabetes'})

int_columns = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI',
    'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 
    'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'Diabetes'
]

# Convert specified columns to int
training_data[int_columns] = training_data[int_columns].astype(int)

# Ensure 'BMI' remains as float
training_data['BMI'] = training_data['BMI'].astype(float)
#print(training_data.info())

conn = sqlite3.connect('diabetes_data.db')
c = conn.cursor()

# Create a table for storing user inputs
c.execute('''
CREATE TABLE IF NOT EXISTS user_diabetes_prediction_data (
    id INTEGER PRIMARY KEY,
    HighBP INTEGER,
    HighChol INTEGER,
    CholCheck INTEGER,
    BMI REAL,
    Smoker INTEGER,
    Stroke INTEGER,
    HeartDiseaseorAttack INTEGER,
    PhysActivity INTEGER,
    Fruits INTEGER,
    Veggies INTEGER,
    HvyAlcoholConsump INTEGER,
    AnyHealthcare INTEGER,
    NoDocbcCost INTEGER,
    GenHlth INTEGER,
    MentHlth INTEGER,
    PhysHlth INTEGER,
    DiffWalk INTEGER,
    Sex INTEGER,
    Age INTEGER,
    Education INTEGER,
    Income INTEGER,
    prediction TEXT
)
''')

# Train the model
x = training_data.drop("Diabetes", axis=1)
y = training_data["Diabetes"]

"""training feature: 80%, testing feature 20%, training label: 80%, testing leabel: 20%"""
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.2, random_state=42)

#Logistic regression for linear relation
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

def insert_prediction_data(features, prediction):
    c.execute("""
    Insert INTO user_diabetes_prediction_data (
        'HighBP', 'HighChol', 'CholCheck', 'BMI',
        'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 
        'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
        'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
        'DiffWalk', 'Sex', 'Age', 'Education', 'Income', 'prediction'
        )VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (*features, prediction))
    conn. commit()


"""lack of a interface to input the data from users"""
def enter_user_data():
    # Create a list to store the features
    features = []

    # Gather user input through simpledialog
    try:
        HighBP = int(simpledialog.askstring("Input", "Have you ever been told by a doctor that you have high blood pressure (0 for No, 1 for Yes)?"))
        features.append(HighBP)

        HighChol = int(simpledialog.askstring("Input", "Have you ever been told by a doctor that you have high cholesterol (0 for No, 1 for Yes)?"))
        features.append(HighChol)

        CholCheck = int(simpledialog.askstring("Input", "Have you had a cholesterol check in the past 5 years (0 for No, 1 for Yes)?"))
        features.append(CholCheck) 

        BMI = float(simpledialog.askstring("Input", "Input your BMI in 1 decimal place (weight in kg / (height in m)^2):"))
        features.append(BMI)

        Smoker = int(simpledialog.askstring("Input", "Have you smoked at least 5 packs of cigarettes in your entire life (0 for No, 1 for Yes)?"))
        features.append(Smoker)

        Stroke = int(simpledialog.askstring("Input", "Have you ever had a stroke (0 for No, 1 for Yes)?"))
        features.append(Stroke)

        HeartDiseasorAttack = int(simpledialog.askstring("Input", "Do you have coronary heart disease or myocardial infarction (0 for No, 1 for Yes)?"))
        features.append(HeartDiseasorAttack)

        PhysActivity = int(simpledialog.askstring("Input", "Do you do physical activity in the past 30 days (0 for No, 1 for Yes)?"))
        features.append(PhysActivity)

        Fruits = int(simpledialog.askstring("Input", "Do you consume fruits one or more times per day (0 for No, 1 for Yes)?"))
        features.append(Fruits)

        Veggies = int(simpledialog.askstring("Input", "Do you consume vegetables one or more times per day (0 for No, 1 for Yes)?"))
        features.append(Veggies)

        HvyAlcoholConsump = int(simpledialog.askstring("Input", "Do you have more than 14 drinks of alcohol per week for men and more than 7 drinks of alcohol per week for women (0 for No, 1 for Yes)?"))
        features.append(HvyAlcoholConsump)

        AnyHealthcare = int(simpledialog.askstring("Input", "Do you have any kind of health care coverage (0 for No, 1 for Yes)?"))
        features.append(AnyHealthcare)

        NoDocbcCost = int(simpledialog.askstring("Input", "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost (0 for No, 1 for Yes)?"))
        features.append(NoDocbcCost)

        GenHlth = int(simpledialog.askstring("Input", "In general, how would you rate your health (1-5, where 1 is worst and 5 is best)?"))
        features.append(GenHlth)

        MenHlth = int(simpledialog.askstring("Input", "How many days during the past 30 days was your mental health not good (0-30)?"))
        features.append(MenHlth)

        PhysHlth = int(simpledialog.askstring("Input", "How many days during the past 30 days was your physical health not good (0-30)?"))
        features.append(PhysHlth)

        DiffWalk = int(simpledialog.askstring("Input", "Do you have difficulty walking or climbing stairs (0 for No, 1 for Yes)?"))
        features.append(DiffWalk)

        Sex = int(simpledialog.askstring("Input", "Input your gender (0 for Female, 1 for Male):"))
        features.append(Sex)

        Age = int(simpledialog.askstring("Input", "Input your age group by number (1-14): (1 = 18 - 24 / 2 = 25 - 29/ 3 = 30 - 34 / 4 = 35 - 39 / 5 = 40 - 44 / 6 = 45 - 49 / 7 = 50 - 54 / 8 = 55 - 59 / 9 = 60 - 64 / 10 = 65 - 69 / 11 = 70 - 74 / 12 = 75 - 79 / 13 = 80 or older / 14 = don’t know/refused "))
        features.append(Age)

        Education = int(simpledialog.askstring("Input", "Input your education level by number (1-6) (1 = Never attended school or only kindergarten 2 = primary school  / 3 = secondary school / 4 = DSE / 5 = pre graduation of college / 6 = College graduate) :"))
        features.append(Education)

        Income = int(simpledialog.askstring("Input", "Input your income level by number (1-8) 1 = Less than $10000 / 2 = Less than $15000 / 3 = Less than $20000 / 4 = Less than $25000 / 5 = Less than $30000 / 6 = Less than $50000 / 7 = Less than $75000 / 8 = $75000 or more) :"))
        features.append(Income)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return None

    # Convert the features to a NumPy array and reshape for prediction
    feature_array = np.array(features).reshape(1, -1)

    # Make the prediction
    prediction = lr.predict(feature_array)[0]
    prediction_text = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    # Insert data into the SQLlite database
    insert_prediction_data(features, prediction_text)

    return prediction_text

# Function to handle the prediction process
def predict_diabetes():
    result = enter_user_data()
    if result is not None:
        messagebox.showinfo("Prediction Result", f"The prediction is {result}\nThank you for using the diabetes prediction tool!")
        messagebox.showinfo("Disclaimer", f"This information is for reference only and does not replace professional medical advice. Consult your doctor with any questions.")

# Function to present the correlation table
def present_correlation_table(): 
    correlation_table = create_correlation_table()

    if correlation_table is not None:
        messagebox.showinfo("Correlation Table", str(correlation_table))
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")

def present_diabetes_pyramid():
    diabetes_pyramid = create_diabetes_pyramid()

    if diabetes_pyramid is not None:
        messagebox.showinfo("Diabetes pyramid", str(diabetes_pyramid))
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")

def present_genhlth_bmi_diabetes():
    genhlth_bmi_diabetes = create_genhlth_bmi_diabetes()

    if genhlth_bmi_diabetes is not None:
        messagebox.showinfo("Genhlth BMI diabetes Graph", str(genhlth_bmi_diabetes))
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")

def present_smoker_hvyalcoholconsump_diabetes():
    smoker_hvyalcoholconsump_diabetes = create_smoker_hvyalcoholconsump_diabetes()
    
    if smoker_hvyalcoholconsump_diabetes is not None:
        messagebox.showinfo("Smoker hvyalcoholconsump diabetes Graph", str(smoker_hvyalcoholconsump_diabetes))
        messagebox.showwarning("Warning", "Graph has been shown.")
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")

def present_education_income_diabetes():
    education_income_diabetes = create_education_income_diabetes()
   
    if education_income_diabetes is not None:
        messagebox.showinfo("Education income diabetes Graph", str(education_income_diabetes))
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")     

def present_create_highBP_diabetes():
    highBP_diabetes = create_highBP_diabetes()
   
    if highBP_diabetes is not None:
        messagebox.showinfo("HighBP diabetes Graph", str(highBP_diabetes))
    else:
        messagebox.showwarning("Warning", "Graph has been shown.")       

# Create the main application window
def create_gui():
    root = tk.Tk()
    root.geometry("400x400") # define the window width and height
    root.configure(background="cadetblue1")
    root.title("Diabetes Prediction")

    input_button = tk.Button(root, text="Input Data and Predict", command=predict_diabetes)
    input_button.pack(pady=20)

    button = tk.Button(root, text="Show Correlation Table", command=present_correlation_table)
    button.pack(pady=20)

    button = tk.Button(root, text="Show Diabetes Pyramid", command=present_diabetes_pyramid)
    button.pack(pady=20)

    button = tk.Button(root, text="Show Genhlth BMI Diabetes Graph", command=present_genhlth_bmi_diabetes)
    button.pack(pady=20)
    
    button = tk.Button(root, text="Show Smoker Hvyalcoholconsump Diabetes Graph", command=present_smoker_hvyalcoholconsump_diabetes)
    button.pack(pady=20)

    button = tk.Button(root, text="Show Education Income Diabetes Graph", command=present_education_income_diabetes)
    button.pack(pady=20)

    button = tk.Button(root, text="Show HighBP Diabetes Graph", command=present_create_highBP_diabetes)
    button.pack(pady=20)

    root.mainloop()


create_gui()
conn.close()

