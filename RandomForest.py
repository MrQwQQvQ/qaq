# Used to ignore warnings
import warnings
warnings.filterwarnings("ignore")

import collections
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Read data
file_path = 'data.csv'
df=pd.read_csv(file_path)

# Round the data
df = df.applymap(lambda x: round(x, 0))

# Modify abnormal BMI values
mask = df['BMI'] > 50
df.loc[mask, 'BMI'] = 50

# Separate features and labels, remove 'id' column
X = df.iloc[:, 1:-1]
Y = df.iloc[:, -1]

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2333)

# Generate duplicates for target=1 or 2 data
mask = (Y_train == 1)
X_train_dup = X_train[mask]
Y_train_dup = Y_train[mask]
for i in range(1, 4):
    X_train = pd.concat([X_train, X_train_dup], ignore_index=True)
    Y_train = pd.concat([Y_train, Y_train_dup], ignore_index=True)

mask = (Y_train == 2)
X_train_dup = X_train[mask]
Y_train_dup = Y_train[mask]
for i in range(1, 3):
    X_train = pd.concat([X_train, X_train_dup], ignore_index=True)
    Y_train = pd.concat([Y_train, Y_train_dup], ignore_index=True)

print(Y_train.value_counts())

# Generate model and fit
rfc = RandomForestClassifier(
    random_state=2333,
    # n_estimators=100,
    # verbose=1,
)
rfc.fit(X_train, Y_train)
Y_pred0 = rfc.predict(X_train)

# Generate fitting results
print(collections.Counter(Y_pred0))
f1 = f1_score(Y_train, Y_pred0, average='macro')
print(f"F1 Score of train data: {f1}")
print(classification_report(Y_train, Y_pred0))

Y_pred = rfc.predict(X_test)

print(collections.Counter(Y_pred))

f1 = f1_score(Y_test, Y_pred, average='macro')
print(f"F1 Score of test data: {f1}")
print(classification_report(Y_test, Y_pred))