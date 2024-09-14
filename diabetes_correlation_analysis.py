import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# print(df.info())

"""Print the number of diabetes and non diabetes"""
# count_of_ones = df["Diabetes_binary"].value_counts().get(1, 0)
# count_of_ones = df["Diabetes_binary"].value_counts().get(0, 1)
# print(count_of_ones)

def create_correlation_table():
    """Analysis 1: Correlation table"""

    # Calculate correlation with diabetes and drop specified columns
    columns_to_drop = []
    if 'ID' in df.columns:
        columns_to_drop.append('ID')
    if 'Diabetes_binary' in df.columns:
        columns_to_drop.append('Diabetes_binary')
    
    correlation_with_diabetes = df.corr()["Diabetes_binary"].drop(columns_to_drop)  # Remove 'ID' and 'Diabetes_binary' to avoid self-correlation
    correlation_table_of_diabetes = correlation_with_diabetes.reset_index()  # Change Series to DataFrame
    correlation_table_of_diabetes.columns = ["Factor", "Correlation_with_diabetes"]  # Rename columns
    correlation_table_of_diabetes = correlation_table_of_diabetes.sort_values("Correlation_with_diabetes", ascending=False).reset_index(drop=True)

    print(correlation_table_of_diabetes)

    #Export to Excel
    export_excel = correlation_table_of_diabetes.copy()
    export_excel.index = range(1, len(export_excel) + 1)  # Start index from 1
    export_excel.to_excel("correlation_table_of_diabetes.xlsx", index=True)

    #Create a figure and axis
    fig, ax = plt.subplots()
    
    # Hide the axes
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=correlation_table_of_diabetes.values, colLabels=correlation_table_of_diabetes.columns, loc='center')
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(horizontalalignment='center', verticalalignment='center')
    
    plt.show()


def create_diabetes_pyramid():
    """Analysis 2: Age, sex & diabetes"""

    df['Sex'] = ['Female' if x == 0 else 'Male' for x in df['Sex']]

    pyramid_data = df[df['Diabetes_binary'] == 1].groupby(['Age', 'Sex']).size().unstack(fill_value=0)

    female_counts = pyramid_data['Female']
    male_counts = pyramid_data['Male']
    
    plt.figure(figsize=(10, 6))

    # Plot females on the left
    plt.barh(female_counts.index, -female_counts, color='salmon', label='Female', align='center')
    plt.barh(male_counts.index, male_counts, color='lightblue', label='Male', align='center')
    
    plt.axvline(0, color='black', linewidth=0.8) 
    
    plt.xlabel("Number of People with Diabetes")
    plt.ylabel("Age group")
    plt.title("Incidence of Diabetes, by Age and Sex")
    plt.grid()

    plt.xticks(ticks=[-3000, -2000, -1000, 0, 1000, 2000, 3000],
               labels=["3000", "2000", "1000", "0", "1000", "2000", "3000"])
    
    age_group = ["0", "18 - 24", "25 - 29", "30 - 34", "35 - 39", "40 - 44",
                 "45 - 49", "50 - 54", "55 - 59", "60 - 64", "65 - 69",
                 "70 - 74", "75 - 79", "80 or older"]

    plt.yticks(ticks=range(len(age_group)), labels=age_group)
    
    plt.legend()
    plt.show()

def create_genhlth_bmi_diabetes():
    """Analysis 3: GenHlth, BMI and diabetes"""
    
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")

    sns.catplot(x="GenHlth", y="BMI", data=df, hue="Diabetes_binary", kind="bar", height=6, aspect=1.5)

    plt.title("Relation between GenHlth, BMI and Diabetes", y=1.05) 
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.85, top=0.85, wspace=0.2, hspace=0.2)
    plt.gca().spines['left'].set_visible(False)  # Hide the y-axis spine
    plt.show()

def create_smoker_hvyalcoholconsump_diabetes():
    """Analysis 4: Smoker, HvyAlcoholConsump, diabetes"""
    
    sns.set_style("whitegrid")
    sns.catplot(x="Smoker", y="HvyAlcoholConsump", data=df, hue="Diabetes_binary", kind="bar", height=6, aspect=1.2)

    plt.title("Relation between Smoker, Heavy Alcohol Consumption and Diabetes", y=0.96)
    plt.gca().spines['left'].set_visible(False)  # Hide the y-axis spine
    plt.show()

def create_education_income_diabetes():
    """Analysis 5: Education, Income and diabetes"""

    sns.set_style("whitegrid")
    
    sns.catplot(x="Education", y="Income", data=df, hue="Diabetes_binary", kind="bar", height=6, aspect=1.8)
    
    education_level = ["No School / Kindergarten", "Primary school", "Secondary school", 
                       "DSE", "Pre graduation of college", "College graduate"]
    plt.xticks(ticks=range(len(education_level)), labels=education_level)
    
    Income  = ["Less than $10000", "Less than $15000", "Less than $20000", "Less than $25000",
               "Less than $30000", "Less than $50000", "Less than $75000", "$75000 or more"]
    plt.yticks(ticks=range(len(Income)), labels=Income)

    plt.title("Relation between Education, Income and Diabetes", y=0.96)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
    plt.gca().spines['left'].set_visible(False)  # Hide the y-axis spine
    plt.show()


def create_highBP_diabetes():
    """Analysis 6: HighBP and diabetes"""
    # Create a count plot for HighBP and Diabetes
    
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    filtered_df = df[df['Diabetes_binary'] == 1]
    
    sns.countplot(x="HighBP", data=filtered_df)
    
    plt.xlabel("HighBP")
    plt.ylabel("Number of people")
    plt.title("Distribution of HighBP Among Diabetic Individuals", y=1.02)
    
    plt.gca().spines['left'].set_visible(False)  # Hide the y-axis spine
    plt.gca().spines['right'].set_visible(False) # Hide the right spine
    plt.gca().spines['top'].set_visible(False)  # Hide the upper spine
    plt.show()


# create_correlation_table()
# create_diabetes_pyramid()
# create_genhlth_bmi_diabetes()
# create_smoker_hvyalcoholconsump_diabetes()
# create_education_income_diabetes()
# create_highBP_diabetes()