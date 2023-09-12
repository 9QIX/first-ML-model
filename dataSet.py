import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# Define the features (independent variables)
x = df.drop('logS', axis=1)

# Define the target (dependent variable)
y = df['logS']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Create a Linear Regression model
lr = LinearRegression()

# Fit the model to the training data
lr.fit(x_train, y_train)

# Predictions on the training and testing sets
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Calculate Mean Squared Error and R-squared for training set
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# Calculate Mean Squared Error and R-squared for testing set
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print the results
print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)

# Create a DataFrame to store the results
lr_results = pd.DataFrame({'Method': ['Linear Regression'],
                            'Training MSE': [lr_train_mse],
                            'Training R2': [lr_train_r2],
                            'Test MSE': [lr_test_mse],
                            'Test R2': [lr_test_r2]})

# Display the results DataFrame
print(lr_results)
