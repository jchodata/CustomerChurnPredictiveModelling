import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle


# Import data into a DataFrame.
PATH = "D:\\data\\"
FILE = "CustomerChurn_Mystery.csv"

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

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.
X = df.copy()  # Create a separate copy to prevent unwanted tampering with data.

del X['CustomerID']  # Delete CustomerID which is completely random.

# Adding an intercept *** This is required ***. Don't forget this step.
X = sm.add_constant(X)

# Assigning binary values to columns that only contain yes or no values
yesNoColumns = ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled']

for col in yesNoColumns:
    X[col] = X[col].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)

X = pd.get_dummies(X, columns=['SubscriptionType', 'PaymentMethod', 'ContentType',
                               'DeviceRegistered', 'GenrePreference', 'Gender'], dtype=int)

# load the column names
with open("column_names.pkl", "rb") as file:
    loaded_column_names = pickle.load(file)

# Subset X with the top features
X_top19 = X[loaded_column_names]

# Load X scaler from file.
scalerXFile = open("scalerX.pkl",'rb')
loadedXScaler = pickle.load(scalerXFile)
scalerXFile.close()

# Load y scaler from file.
scalerYFile = open("scalerY.pkl",'rb')
loadedYScaler = pickle.load(scalerYFile)
scalerYFile.close()

# Load pre-trained model.
file = open("bestModel.pkl", 'rb')
loadedModel = pickle.load(file)
file.close()

# Access features from the loaded model.
loaded_features = loadedModel['feature_names']
print("Loaded Features:", loaded_features)

# Access scaler and model from the loaded dictionaries.
loadedXScaler = loadedXScaler['scaler']
loadedModel = loadedModel['model']

# Make predictions
predictions = loadedModel.predict(X_top19)

print(predictions)

# Store predictions in a dataframe
dfPredictions = pd.DataFrame()
listPredictions = []
for i in range(0, len(predictions)):
    prediction = predictions.flatten()[i]
    listPredictions.append(prediction)
dfPredictions['Predictions'] = listPredictions

dfPredictions.to_csv('Predictions.csv', index=False)
