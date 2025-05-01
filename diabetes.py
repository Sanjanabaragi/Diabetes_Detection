import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Display the first few rows and statistical summary of the data
print(df.head())  
print(df.describe())  

# Replace zero values with NaN in specified columns and fill NaN values with the mean of each column
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)  

# Split the data into features and target
X = df.drop(columns='Outcome')  
y = df['Outcome']  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Function to get user input
def get_user_input():
    gender = input("Enter your gender (male/female): ").strip().lower()
    
    if gender == "female":
        pregnancies = float(input("Enter the number of pregnancies: "))
    else:
        pregnancies = 0  # Set pregnancies to 0 for males

    glucose = float(input("Enter glucose level: "))
    blood_pressure = float(input("Enter blood pressure level: "))
    skin_thickness = float(input("Enter skin thickness: "))
    insulin = float(input("Enter insulin level: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree_function = float(input("Enter diabetes pedigree function: "))
    age = float(input("Enter age: "))

    # Combine user inputs into a DataFrame with column names matching the training data
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                               bmi, diabetes_pedigree_function, age]], 
                               columns=X.columns)

    return user_data

# Get user input and make predictions
user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)

# Output the prediction result
if prediction == 1:
    print("The model predicts: You have a risk of diabetes.")
else:
    print("The model predicts: You are not at risk of diabetes.")