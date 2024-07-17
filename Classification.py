import warnings
import pandas as pd
import numpy as np
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

df = pd.read_csv('megastore-classification-dataset.csv')

X = df.loc[:, df.columns != 'ReturnCategory']
y = df[['ReturnCategory']]

# checking if there are any empty values or nulls:
X.info()  # No Nulls were found.
y.info()  # No Nulls were found.

# Checking of the presence of duplicates:
duplicates_x = X.duplicated()
num_duplicates = duplicates_x.sum()  # Output: 0
print("number of duplicates in X:", num_duplicates)

# save values to fill nulls
c = ['Sales', 'Discount']
for i in c:
    mean_value = df[i].mean()
    pickle.dump(mean_value, open(f'{i}C.pickle', 'wb'))
    X[i] = X[i].fillna(mean_value)

pickle.dump(df['Quantity'].median(), open('QuantityC.pickle', 'wb'))

mode = df[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID', 'CategoryTree',
           'Order Date', 'Ship Date']].mode()

pickle.dump(mode, open(f'modeC.pickle', 'wb'))

# Dropping unnecessary columns:
X.drop('Row ID', axis=1, inplace=True)
X.drop('Order ID', axis=1, inplace=True)
X.drop('Customer ID', axis=1, inplace=True)
X.drop('Customer Name', axis=1, inplace=True)
X.drop('Product Name', axis=1, inplace=True)
X.drop('Postal Code', axis=1, inplace=True)

# Processing Dates:
# creating "Order Month" && "Order Year" feature from 'Order Date' column:
X['Order Date'] = pd.to_datetime(X['Order Date'])
X['Order Month'] = X['Order Date'].dt.strftime('%m')
X['Order Year'] = X['Order Date'].dt.strftime('%Y')

X.drop('Order Date', axis=1, inplace=True)

# creating "Ship Month" && "Ship Year" feature from 'Ship Date' column:
X['Ship Date'] = pd.to_datetime(X['Ship Date'])
X['Ship Month'] = X['Ship Date'].dt.strftime('%m')
X['Ship Year'] = X['Ship Date'].dt.strftime('%Y')

X.drop('Ship Date', axis=1, inplace=True)

# Processing 'CategoryTree' column:
df_normalized = pd.json_normalize(X['CategoryTree'].apply(eval))
X['Main Category'] = df_normalized['MainCategory']
X['Sub Category'] = df_normalized['SubCategory']

X.drop('CategoryTree', axis=1, inplace=True)


# product number
def extract_features(string):
    split_string = string.split('-')
    feature3 = split_string[2]
    return feature3


X['Product ID Number'] = X['Product ID'].apply(extract_features)
X.drop('Product ID', axis=1, inplace=True)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding:
ordinal = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=13)
X_train['Order Month'] = ordinal.fit_transform(np.array(X_train['Order Month']).reshape(-1, 1))
X_test['Order Month'] = ordinal.transform(np.array(X_test['Order Month']).reshape(-1, 1))
pickle.dump(ordinal, open('Order MonthC.pickle', 'wb'))

ordinal1 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Order Year'] = ordinal1.fit_transform(np.array(X_train['Order Year']).reshape(-1, 1))
X_test['Order Year'] = ordinal1.transform(np.array(X_test['Order Year']).reshape(-1, 1))
pickle.dump(ordinal1, open('Order YearC.pickle', 'wb'))

ordinal2 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=13)
X_train['Ship Month'] = ordinal2.fit_transform(np.array(X_train['Ship Month']).reshape(-1, 1))
X_test['Ship Month'] = ordinal2.transform(np.array(X_test['Ship Month']).reshape(-1, 1))
pickle.dump(ordinal2, open('Ship MonthC.pickle', 'wb'))

ordinal3 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=6)
X_train['Ship Year'] = ordinal3.fit_transform(np.array(X_train['Ship Year']).reshape(-1, 1))
X_test['Ship Year'] = ordinal3.transform(np.array(X_test['Ship Year']).reshape(-1, 1))
pickle.dump(ordinal3, open('Ship YearC.pickle', 'wb'))

ordinal4 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Ship Mode'] = ordinal4.fit_transform(np.array(X_train['Ship Mode']).reshape(-1, 1))
X_test['Ship Mode'] = ordinal4.transform(np.array(X_test['Ship Mode']).reshape(-1, 1))
pickle.dump(ordinal4, open('Ship ModeC.pickle', 'wb'))

ordinal5 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=4)
X_train['Segment'] = ordinal5.fit_transform(np.array(X_train['Segment']).reshape(-1, 1))
X_test['Segment'] = ordinal5.transform(np.array(X_test['Segment']).reshape(-1, 1))
pickle.dump(ordinal5, open('SegmentC.pickle', 'wb'))

ordinal6 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=2)
X_train['Country'] = ordinal6.fit_transform(np.array(X_train['Country']).reshape(-1, 1))
X_test['Country'] = ordinal6.transform(np.array(X_test['Country']).reshape(-1, 1))
pickle.dump(ordinal6, open('CountryC.pickle', 'wb'))

ordinal7 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=50)
X_train['State'] = ordinal7.fit_transform(np.array(X_train['State']).reshape(-1, 1))
X_test['State'] = ordinal7.transform(np.array(X_test['State']).reshape(-1, 1))
pickle.dump(ordinal7, open('StateC.pickle', 'wb'))

ordinal8 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Region'] = ordinal8.fit_transform(np.array(X_train['Region']).reshape(-1, 1))
X_test['Region'] = ordinal8.transform(np.array(X_test['Region']).reshape(-1, 1))
pickle.dump(ordinal8, open('RegionC.pickle', 'wb'))

ordinal9 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=4)
X_train['Main Category'] = ordinal9.fit_transform(np.array(X_train['Main Category']).reshape(-1, 1))
X_test['Main Category'] = ordinal9.transform(np.array(X_test['Main Category']).reshape(-1, 1))
pickle.dump(ordinal9, open('Main CategoryC.pickle', 'wb'))

ordinal10 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=18)
X_train['Sub Category'] = ordinal10.fit_transform(np.array(X_train['Sub Category']).reshape(-1, 1))
X_test['Sub Category'] = ordinal10.transform(np.array(X_test['Sub Category']).reshape(-1, 1))
pickle.dump(ordinal10, open('Sub CategoryC.pickle', 'wb'))

ordinal11 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=500)
X_train['City'] = ordinal11.fit_transform(np.array(X_train['City']).reshape(-1, 1))
X_test['City'] = ordinal11.transform(np.array(X_test['City']).reshape(-1, 1))
pickle.dump(ordinal11, open('cityC.pickle', 'wb'))

ordinal12 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=1541)
X_train['Product ID Number'] = ordinal12.fit_transform(np.array(X_train['Product ID Number']).reshape(-1, 1))
X_test['Product ID Number'] = ordinal12.transform(np.array(X_test['Product ID Number']).reshape(-1, 1))
pickle.dump(ordinal12, open('Product ID NumberC.pickle', 'wb'))

#target encoding
ordinal13 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=1541)
y_train['ReturnCategory'] = ordinal13.fit_transform(np.array(y_train['ReturnCategory']).reshape(-1, 1))
y_test['ReturnCategory'] = ordinal13.transform(np.array(y_test['ReturnCategory']).reshape(-1, 1))
pickle.dump(ordinal13, open('ReturnCategoryC.pickle', 'wb'))


# Scaling:
# using MinMax
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

pickle.dump(scaler, open('scalerrrC.pickle', 'wb'))

"""
# using standard
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
"""

# Feature Selection:
# Training set:
# Categorical Methods:
categorical_features = X_train[
    ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID Number', 'Main Category', 'Sub Category',
     'Order Month', 'Order Year', 'Ship Month', 'Ship Year']]

# Chi-Squared
selector1 = SelectKBest(score_func=chi2, k=5)
selected_categorical_features = selector1.fit_transform(categorical_features, y_train['ReturnCategory'])
new_df1 = pd.DataFrame(selected_categorical_features,
                       columns=categorical_features.columns[selector1.get_support(indices=True)])
print(f"indices of selected features using Chi-Squared: {selector1.get_support(indices=True)}")

# Mutual Information
selector2 = SelectKBest(score_func=mutual_info_classif, k=5)
selected_categorical_features2 = selector2.fit_transform(categorical_features, y_train['ReturnCategory'])

print(f"indices of selected features using Mutual Info: {selector2.get_support(indices=True)}")

# Numerical Methods:
numerical_features = X_train[['Sales', 'Quantity', 'Discount']]

# Anova
selector3 = SelectKBest(score_func=f_classif, k=2)
selected_numerical_features = selector3.fit_transform(numerical_features, y_train['ReturnCategory'])
print(f"indices of selected features using Anova: {selector3.get_support(indices=True)}")

# Kendall's
numerical_features["ReturnCategory"] = y_train["ReturnCategory"].values

corr = numerical_features.corr(method="kendall")
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.title("kendall's")
plt.show()

selected_features = corr.index[abs(corr["ReturnCategory"]) > 0.02]
new_df2 = numerical_features[selected_features.delete(-1)]

X_train_new = pd.concat([new_df1, new_df2], axis=1, join="inner")

print("X_train_new-------------------------------")
X_train_new.info()

# feature selection on test set:
cat_features = X_test[
    ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID Number', 'Main Category',
     'Sub Category', 'Order Month', 'Order Year', 'Ship Month', 'Ship Year']]

X_test_cat = cat_features.iloc[:, selector1.get_support(indices=True)]
X_test_num = X_test[selected_features.delete(-1)]
X_test_new = pd.concat([X_test_cat, X_test_num], axis=1, join="inner")

print("X_test_new-------------------------------")
X_test_new.info()

print("----------------- CLASSIFICATION MODELS -------------------------")


def classification_model(algo):
    print(f"---------------- {algo} Model --------------")
    if algo == 'Logistic regression':
        model = LogisticRegression(random_state=42, multi_class='multinomial')
        param_grid = {'penalty': ['l1', 'l2'], 'C': [0.2, 1, 200]}  # Define the hyperparameters to search over
    elif algo == 'Desicion Trees':
        model = DecisionTreeClassifier(random_state=42, criterion='entropy')
        param_grid = {'max_depth': [10, 12, 14], 'min_samples_split': [10, 30, 50]}
    elif algo == 'Random Forest':
        model = RandomForestClassifier(random_state=42, criterion='entropy')
        param_grid = {'n_estimators': [100, 60, 55], 'min_samples_leaf': [1, 2, 4]}
    elif algo == 'svm':
        model = svm.SVC(random_state=42, kernel='linear')
        param_grid = {'C': [30, 50, 10], 'max_iter': [-1, 5, 10]}

    # accuracy without tuning
    model.fit(X_train_new, y_train)
    pred = model.predict(X_test_new)
    # Create a grid search object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    # Fit the grid search object to the training data
    train_start_time = time.time()
    grid_search.fit(X_train_new, y_train)
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    # model saving
    if algo == 'Desicion Trees':
        pickle.dump(grid_search.best_estimator_, open("dtc.pickle", 'wb'))

    # Print the best hyperparameters and model performance
    print(f"Best parameters for {algo}: {grid_search.best_params_}")
    print(f"mean test score for cv of {algo}: ", grid_search.cv_results_['mean_test_score'])
    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    # Make predictions on test data
    test_start_time = time.time()
    y_pred = best_estimator.predict(X_test_new)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Total train time for {algo}:{total_train_time:.4f}\nTotal test time for {algo}:{test_time:.4f}\n")
    # Evaluate the model performance on test data
    print(f"Best cross validation score for {algo}: {grid_search.best_score_*100:.2f}% ")
    print(f"Accuracy before tuning for {algo}: {accuracy_score(y_test, pred)*100:.2f}%")
    print(f"Accuracy after tuning for {algo}: {accuracy_score(y_test, y_pred)*100:.2f}%\n")


classification_model('Logistic regression')
classification_model('Desicion Trees')
classification_model('Random Forest')
classification_model('svm')

