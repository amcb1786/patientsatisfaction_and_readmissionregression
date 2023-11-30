#!/usr/bin/env python
# coding: utf-8

# In[304]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[305]:


#Reading the file
filepath = 'Week14Assignment.txt'
df = pd.read_csv(filepath)
for column in df.columns:
    print(f"{column}: {df[column].unique()}")


# In[306]:


df.columns = df.columns.str.strip()
print(df.columns)


# In[307]:


#The number of readmitted patients 
readmitted_count = df['Readmission'].sum()
print(f"Number of patients readmitted: {readmitted_count}")


# In[308]:


#Calculate average satisfaction score for each category
satisfaction_categories = ['StaffSatisfaction', 'CleanlinessSatisfaction', 'FoodSatisfaction', 'ComfortSatisfaction', 'CommunicationSatisfaction']
for category in satisfaction_categories:
    avg_score = df[category].mean()
    print(f"Average satisfaction score for {category}: {avg_score}")


# In[309]:


#Calculate overall satisfaction
df['OverallSatisfaction'] = df[satisfaction_categories].mean(axis=1)

#Plot overall satisfaction
plt.figure(figsize=(10, 6))
plt.plot(df['PatientID'], df['OverallSatisfaction'], color='green')
plt.xlabel('Patient ID')
plt.ylabel('Overall Satisfaction')
plt.title('Overall Satisfaction of Patients')
plt.show()


# In[310]:


#Logistic Regression
X = df[['StaffSatisfaction', 'CleanlinessSatisfaction', 'FoodSatisfaction', 'ComfortSatisfaction', 'CommunicationSatisfaction']]
y = df['Readmission']
log_reg = LogisticRegression().fit(X, y)


# In[311]:


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Display logistic regression results
print(f"Coefficient: {model.coef_[0][0]}")
print(f"Intercept: {model.intercept_[0]}")

#Check the sign and magnitude of the coefficient
correlation_coefficient = model.coef_[0][0]

#Check correlation and print statements based on the correlation strength
if correlation_coefficient > 0:
    print("There is a positive correlation between Overall Satisfaction and Readmission.")
elif correlation_coefficient < 0:
    print("There is a negative correlation between Overall Satisfaction and Readmission.")
else:
    print("There is little to no correlation between Overall Satisfaction and Readmission.")

#Plot data points and logistic regression curve
plt.figure(figsize=(10, 6))
# ... (your plotting code here)
plt.show()


# In[312]:


#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")


# In[313]:


#Assuming 'OverallSatisfaction' is the feature and 'Readmission' is the target
X = df[['OverallSatisfaction']]
y = df['Readmission']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Display logistic regression results
print(f"Coefficient: {model.coef_[0][0]}")
print(f"Intercept: {model.intercept_[0]}")

#Predict probabilities for the test set
probs = model.predict_proba(X_test)[:, 1]

#Sort the values for proper plotting
sorted_indices = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.values[sorted_indices]
probs_sorted = probs[sorted_indices]
y_test_sorted = y_test.values[sorted_indices]

#Plot the logistic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted, y_test_sorted, color='blue', label='Actual Data Points')
plt.plot(X_test_sorted, probs_sorted, color='red', linewidth=3, label='Logistic Regression Curve')
plt.xlabel('Overall Satisfaction')
plt.ylabel('Readmission Probability')
plt.title('Logistic Regression: Overall Satisfaction vs Readmission')
plt.legend()
plt.show()

