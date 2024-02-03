#import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')


#Preparing/Checking Data
#===================================

# data size
print(f'The dataset contains {df.shape[0]} rows, and {df.shape[1]} columns')

# Always good to check the names of the columns
df.columns

# and check the data types
df.info()

# for every column
for i in df.columns:
    # print how many features it has
    print(i,len(df[i].unique()))

df.describe()

df['A_Winner'] = ''  # Initialize the column with empty strings

#Makes the actual results clearer/easier to understand
for index, row in df.iterrows():

    if row['Winner'] == 1:
        df.at[index, 'A_Winner'] = row['R_Stance']

    elif row['Winner'] == 0:
        df.at[index, 'A_Winner'] = row['B_Stance']
    
    else:
        #Making sure that the data is correct and the previous function worked
        df.at[index, 'A_Winner'] = "Error"

# Save the DataFrame to a new CSV file
df.to_csv('output.csv', index=False)



#Graphs
#===================================


#Graph for age
numeric_columns = df.select_dtypes(include=[np.number])  # Select only numeric columns
quantiles = numeric_columns.quantile([0.1, 0.25, 0.5, 0.75], axis=0)
df['Winner'].value_counts()

sns.set_style('darkgrid')
sns.set_palette('Set2')
# first we make a copy of the dataset to decode the variables (for visualisation␣purposes)
dfC = df.copy()

# Function to determine stances
def stances(row):
    if (row["R_Stance"] == 'Orthodox' and row['Winner'] == 1 or row["B_Stance"] == 'Orthodox' and row['Winner'] == 0):
        return 'Orthodox'
    else:
        return 'Southpaw'

# Function to determine the age condition
def changeW(row):
    if (row['R_age'] >= 30) or (row['B_age'] >= 30):
        return 'Over 30'
    else:
        return 'Under 30'

# Update the 'Winner' column based on the 'Orthodox' condition
df['Winner'] = df['Winner'].apply(lambda x: 'Orthodox' if x == "Orthodox" else 'Southpaw')

# Update the 'B_Stance' and 'R_Stance' columns based on the 'Winner' condition
df['B_Stance'] = df.apply(lambda row: stances(row['B_Stance']) if row['Winner'] == 'Orthodox' else row['B_Stance'], axis=1)
df['R_Stance'] = df.apply(lambda row: stances(row['R_Stance']) if row['Winner'] == 'Orthodox' else row['R_Stance'], axis=1)

# Apply the 'changeW' function to update the 'Winner' column based on age condition
df['Winner'] = df.apply(changeW, axis=1)

# Create a copy of the DataFrame
dfC = df.copy()

# Plot the data
sns.set(rc={'figure.figsize':(9,7)})
sns.countplot(data=dfC, x='R_Stance', hue='Winner')
plt.title('Southpaw vs Orthodox\n')
plt.show()





#graph for reach

numeric_columns = df.select_dtypes(include=[np.number])  # Select only numeric columns
quantiles = numeric_columns.quantile([0.1, 0.25, 0.5, 0.75], axis=0)
df['Winner'].value_counts()

sns.set_style('darkgrid')
sns.set_palette('Set2')
# first we make a copy of the dataset to decode the variables (for visualisation␣purposes)
dfC = df.copy()

# Function to determine stances
def stances(row):
    if (row["R_Stance"] == 'Orthodox' and row['Winner'] == 1 or row["B_Stance"] == 'Orthodox' and row['Winner'] == 0):
        return 'Orthodox'
    else:
        return 'Southpaw'

# Function to determine the age condition
def changeW(row):
    if (row['R_Reach_cms'] >= row['B_Reach_cms']):
        return 'Reach Advantage'
    else:
        return 'Reach Disadvantage'

# Update the 'Winner' column based on the 'Orthodox' condition
df['Winner'] = df['Winner'].apply(lambda x: 'Orthodox' if x == "Orthodox" else 'Southpaw')

# Update the 'B_Stance' and 'R_Stance' columns based on the 'Winner' condition
df['B_Stance'] = df.apply(lambda row: stances(row['B_Stance']) if row['Winner'] == 'Orthodox' else row['B_Stance'], axis=1)
df['R_Stance'] = df.apply(lambda row: stances(row['R_Stance']) if row['Winner'] == 'Orthodox' else row['R_Stance'], axis=1)

# Apply the 'changeW' function to update the 'Winner' column based on age condition
df['Winner'] = df.apply(changeW, axis=1)

# Create a copy of the DataFrame
dfC = df.copy()

# Plot the data
sns.set(rc={'figure.figsize':(9,7)})
sns.countplot(data=dfC, x='R_Stance', hue='Winner')
plt.title('Southpaw vs Orthodox\n')
plt.show()




#graph for height

numeric_columns = df.select_dtypes(include=[np.number])  # Select only numeric columns
quantiles = numeric_columns.quantile([0.1, 0.25, 0.5, 0.75], axis=0)
df['Winner'].value_counts()

sns.set_style('darkgrid')
sns.set_palette('Set2')
# first we make a copy of the dataset to decode the variables (for visualisation␣purposes)
dfC = df.copy()

# Function to determine stances
def stances(row):
    if (row["R_Stance"] == 'Orthodox' and row['Winner'] == 1 or row["B_Stance"] == 'Orthodox' and row['Winner'] == 0):
        return 'Orthodox'
    else:
        return 'Southpaw'

# Function to determine the age condition
def changeW(row):
    if (row['R_Height_cms'] >= row['B_Height_cms']):
        return 'Height Advantage'
    else:
        return 'Height Disadvantage'

# Update the 'Winner' column based on the 'Orthodox' condition
df['Winner'] = df['Winner'].apply(lambda x: 'Orthodox' if x == "Orthodox" else 'Southpaw')

# Update the 'B_Stance' and 'R_Stance' columns based on the 'Winner' condition
df['B_Stance'] = df.apply(lambda row: stances(row['B_Stance']) if row['Winner'] == 'Orthodox' else row['B_Stance'], axis=1)
df['R_Stance'] = df.apply(lambda row: stances(row['R_Stance']) if row['Winner'] == 'Orthodox' else row['R_Stance'], axis=1)

# Apply the 'changeW' function to update the 'Winner' column based on age condition
df['Winner'] = df.apply(changeW, axis=1)

# Create a copy of the DataFrame
dfC = df.copy()

# Plot the data
sns.set(rc={'figure.figsize':(9,7)})
sns.countplot(data=dfC, x='R_Stance', hue='Winner')
plt.title('Southpaw vs Orthodox\n')
plt.show()




#graph for Experience

numeric_columns = df.select_dtypes(include=[np.number])  # Select only numeric columns
quantiles = numeric_columns.quantile([0.1, 0.25, 0.5, 0.75], axis=0)
df['Winner'].value_counts()

sns.set_style('darkgrid')
sns.set_palette('Set2')
# first we make a copy of the dataset to decode the variables (for visualisation␣purposes)
dfC = df.copy()

# Function to determine stances
def stances(row):
    if (row["R_Stance"] == 'Orthodox' and row['Winner'] == 1 or row["B_Stance"] == 'Orthodox' and row['Winner'] == 0):
        return 'Orthodox'
    else:
        return 'Southpaw'

# Function to determine the age condition
def changeW(row):
    if (row['R_total_rounds_fought'] >= row['B_total_rounds_fought']):
        return 'More Round Experience'
    else:
        return 'Less Round Experience'

# Update the 'Winner' column based on the 'Orthodox' condition
df['Winner'] = df['Winner'].apply(lambda x: 'Orthodox' if x == "Orthodox" else 'Southpaw')

# Update the 'B_Stance' and 'R_Stance' columns based on the 'Winner' condition
df['B_Stance'] = df.apply(lambda row: stances(row['B_Stance']) if row['Winner'] == 'Orthodox' else row['B_Stance'], axis=1)
df['R_Stance'] = df.apply(lambda row: stances(row['R_Stance']) if row['Winner'] == 'Orthodox' else row['R_Stance'], axis=1)

# Apply the 'changeW' function to update the 'Winner' column based on age condition
df['Winner'] = df.apply(changeW, axis=1)

# Create a copy of the DataFrame
dfC = df.copy()

# Plot the data
sns.set(rc={'figure.figsize':(9,7)})
sns.countplot(data=dfC, x='R_Stance', hue='Winner')
plt.title('Southpaw vs Orthodox\n')
plt.show()



#Decision Tree
#===================================

#This is the decision tree to predict which fighter will win baised off the relevent variables

df['P_Winner'] = ''  # Initialize the column with empty strings




from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset into 'df' here

# Cleaning data/results
columns_to_drop = [
    'B_avg_KD', 'B_avg_opp_KD', 'B_avg_opp_SIG_STR_pct', 'B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_avg_opp_TD_pct',
    'B_avg_SUB_ATT', 'B_avg_opp_SUB_ATT', 'B_avg_REV', 'B_avg_opp_REV', 'B_avg_SIG_STR_att', 'B_avg_SIG_STR_landed',
    'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed',
    'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_opp_TD_att',
    'B_avg_opp_TD_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_opp_HEAD_att', 'B_avg_opp_HEAD_landed',
    'B_avg_BODY_att', 'B_avg_BODY_landed', 'B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_LEG_att',
    'B_avg_LEG_landed', 'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_DISTANCE_att', 'B_avg_DISTANCE_landed',
    'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed',
    'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed',
    'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_CTRL_time(seconds)', 'B_avg_opp_CTRL_time(seconds)',
    'B_total_time_fought(seconds)', 'B_total_title_bouts', 'B_current_win_streak', 'B_current_lose_streak',
    'B_longest_win_streak', 'B_wins', 'B_losses', 'B_draw', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split',
    'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage',
    'R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct', 'R_avg_TD_pct', 'R_avg_opp_TD_pct',
    'R_avg_SUB_ATT', 'R_avg_opp_SUB_ATT', 'R_avg_REV', 'R_avg_opp_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed',
    'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_TOTAL_STR_att', 'R_avg_TOTAL_STR_landed',
    'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed', 'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_opp_TD_att',
    'R_avg_opp_TD_landed', 'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed',
    'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 'R_avg_LEG_att',
    'R_avg_LEG_landed', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed',
    'R_avg_opp_DISTANCE_att', 'R_avg_opp_DISTANCE_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed',
    'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_GROUND_att', 'R_avg_GROUND_landed',
    'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed', 'R_avg_CTRL_time(seconds)', 'R_avg_opp_CTRL_time(seconds)',
    'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_current_win_streak', 'R_current_lose_streak',
    'R_longest_win_streak', 'R_wins', 'R_losses', 'R_draw', 'R_win_by_Decision_Majority', 'R_win_by_Decision_Split',
    'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage', 'Referee'
]

# Remove leading/trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Drop columns
df = df.drop(columns_to_drop, axis=1)

# Drop rows with missing values
df = df.dropna()

# Assuming 'A_Winner' and 'P_Winner' columns are already present

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Define features and target variable
features = ['B_Stance', 'R_Stance', 'B_age', 'R_age', 'B_Reach_cms', 'R_Reach_cms', 'B_total_rounds_fought', 'R_total_rounds_fought']
target = 'A_Winner'

X = df[features]
y = df[target]

# Convert categorical variables to numerical using one-hot encoding for the entire dataset
X_encoded = pd.get_dummies(X)

# Separate rows where 'P_Winner' is blank and where it's not
blank_winner_rows = df[df['P_Winner'] == '']
non_blank_winner_rows = df[df['P_Winner'] != '']

# Train-test split for computing accuracy
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model on the training set
clf.fit(X_train, y_train)

# Make predictions for the test set
predictions = clf.predict(X_test)

# Compute accuracy on the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Decision Tree Accuracy on Test Set: {accuracy * 100:.2f}%")

# Train the model on the entire dataset
clf.fit(X_encoded, y)

# Make predictions for all rows
if not X_encoded.empty:
    all_predictions = clf.predict(X_encoded)

    # Populate 'P_Winner' column with predictions for all rows
    df.loc[X_encoded.index, 'P_Winner'] = all_predictions

# Save the DataFrame to a new CSV file
df.to_csv('output.csv', index=False)





#Random forrest
#===================================


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming df is your DataFrame containing the features and target variable
# X should contain the features, and y should contain the target variable
X = df[['B_Stance', 'B_age', 'B_Reach_cms', 'R_Stance', 'R_age', 'R_Reach_cms', 'B_total_rounds_fought', 'R_total_rounds_fought']]
y = df['A_Winner']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("")
# print("Random Forest:")
# print(round((accuracy*100), 1), "% Accuracy" )


print("Random Forest Accuracy on Test Set:", f"{round(accuracy * 100, 1)}%")


