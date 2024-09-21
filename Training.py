import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

# Import data into a DataFrame.
PATH = "D:\\data\\"
FILE = "CustomerChurn.csv"

df = pd.read_csv(PATH + FILE, sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def imputeNullValues(colName, df):
    # Create two new column names based on the original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn = []
    indicatorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if (np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if it is imputed (1) or not (0).
        if (isImputed):
            indicatorColumn.append(1)
        else:
            indicatorColumn.append(0)

    # Append new columns to the dataframe but always keep the original column.
    df[indicatorColName] = indicatorColumn
    df[imputedColName] = imputedColumn
    del df[colName]  # Drop the column with null values.
    return df

# Imputing columns that have null values
df = imputeNullValues('AccountAge', df)
df = imputeNullValues('ViewingHoursPerWeek', df)
df = imputeNullValues('AverageViewingDuration', df)

# Binning for imp_AccountAge
df['imp_AccountAge'] = pd.cut(df['imp_AccountAge'], bins=10, labels=False)

X = df.copy()  # Create a separate copy to prevent unwanted tampering with data.
del X['Churn']  # Delete the target variable.
del X['CustomerID']  # Delete CustomerID which is completely random.

y = df['Churn']

# Assigning binary values to columns that only contain yes or no values
yesNoColumns = ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled']

for col in yesNoColumns:
    X[col] = X[col].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)

# Creating dummy variables for categorical columns
X = pd.get_dummies(X, columns=['SubscriptionType', 'PaymentMethod', 'ContentType',
                               'DeviceRegistered', 'GenrePreference', 'Gender'], dtype=int)

#  f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

featuresDf = pd.DataFrame()
for i in range(0, len(X.columns)):
    featuresDf = pd.concat([featuresDf, pd.DataFrame([{"feature":X.columns[i], "ffs":ffs[0][i]}])],
                           ignore_index=True)
featuresDf = featuresDf.sort_values(by=['ffs'])

# Select top 19 features based on F-statistic scores
top_features = featuresDf.tail(19)['feature'].tolist()

# Subset X with the top features
X_top19 = X[top_features]

# Adding an intercept *** This is required ***. Don't forget this step.
X_top19 = sm.add_constant(X_top19)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_top19, y, test_size=0.2, random_state=42)

# Scale data using MinMaxScaler
sc_x = MinMaxScaler()
XScaled = sc_x.fit_transform(X_top19)

# Save the column names to a pickle file
column_names = X_top19.columns.tolist()
with open("column_names.pkl", "wb") as file:
    pickle.dump(column_names, file)

# Save x scaler as pickle
scalerXFile = open("scalerX.pkl", "wb")
pickle.dump({'scaler': sc_x, 'feature_names': X_top19.columns.tolist()}, scalerXFile)
scalerXFile.close()

# Implement scaling for y.
y_scaler = MinMaxScaler()
y_trainScaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

# Save y scaler as pickle
scalerYFile = open("scalerY.pkl", "wb")
pickle.dump(y_scaler, scalerYFile)
scalerYFile.close()

# Train and save logistic regression model
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')
logisticModel.fit(X_train, y_train)
filehandler = open("bestModel.pkl", "wb")
pickle.dump(logisticModel, filehandler)
filehandler.close()

# Save logistic regression model and feature names as a dictionary
model_info = {'model': logisticModel, 'feature_names': X_top19.columns.tolist()}
with open("bestModel.pkl", "wb") as file:
    pickle.dump(model_info, file)

# Load pickle for model and feature names
with open("bestModel.pkl", "rb") as file:
    loaded_model_info = pickle.load(file)

loaded_model = loaded_model_info['model']
loaded_feature_names = loaded_model_info['feature_names']

# Load pickle for y scaler.
scalerYFile = open("scalerY.pkl", 'rb')
loadedYScaler = pickle.load(scalerYFile)
scalerYFile.close()

# Transform test set and make predictions
XScaled_Test = sc_x.transform(X_test)
y_pred = loaded_model.predict(XScaled_Test)
y_prob = loaded_model.predict_proba(XScaled_Test)

# Convert predictions back to actual scale with extracted scaler.
predictions = loadedYScaler.inverse_transform(y_pred.reshape(-1, 1))

# prepare cross-validation with three folds for model selection
kfold = KFold(n_splits=3, shuffle=True)
accuracyList = []
precisionList = []
recallList = []
f1List = []
count = 0

for train_index, test_index in kfold.split(X_top19):
    sc_x = MinMaxScaler()

    # Scale data.
    # Only fit on the training data.
    XScaled_Train = sc_x.fit_transform( \
        X_top19.loc[X_top19.index.intersection(train_index), :])

    # Transform the test data.
    XScaled_Test = sc_x.transform( \
        X_top19.loc[X_top19.index.intersection(test_index), :])

    #  y does not need to be scaled since it is 0 or 1.
    y_train, y_test = y[train_index], y[test_index]

    # Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    XScaled_Train_resampled, y_train_resampled = smote.fit_resample(XScaled_Train, y_train)

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True,
                                       solver='liblinear')

    # Fit the logistic model on the resampled training set.
    logisticModel.fit(XScaled_Train_resampled, y_train_resampled)

    y_pred = logisticModel.predict(XScaled_Test)
    y_prob = logisticModel.predict_proba(XScaled_Test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])
    count += 1
    print("\n***K-fold: " + str(count))

    # Calculate accuracy and precision scores and add to the list.
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

    print('\nAccuracy: ', accuracy)
    print("\nPrecision: ", precision)
    print("\nrecall: ", recall)
    print("\nf1: ", f1)
    print("\nConfusion Matrix")
    print(cm)

# Show averages of scores over multiple runs.
print("\nAccuracy and Standard Deviation For All Folds:")
print("*********************************************")
print("Average accuracy:  " + str(np.mean(accuracyList)))
print("Accuracy std:      " + str(np.std(accuracyList)))
print("Average precision: " + str(np.mean(precisionList)))
print("Precision std:     " + str(np.std(precisionList)))
print("Average recall: " + str(np.mean(recallList)))
print("Recall std:     " + str(np.std(recallList)))
print("Average f1: " + str(np.mean(f1List)))
print("F1 std:     " + str(np.std(f1List)))