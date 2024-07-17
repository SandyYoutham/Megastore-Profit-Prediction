# your code goes here
import matplotlib
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pickle

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('megastore-regression-dataset.csv')

X = df.loc[:, df.columns != 'Profit']
y = df[['Profit']]

# OUTLIER DETECTION:
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

t = y.values.tolist()
mean = np.mean(t)
std = np.std(t)
outliers = []
sns.set_theme()
# sns.histplot(data=t).set(title="Distribution of Scores", xlabel="Scores")
for i in t:
    z_score = (i - mean) / std
    if np.abs(z_score) > 3:
        outliers.append(i)
print(f'outliers for the first are {len(outliers)}')

q1, q3 = np.percentile(t, [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
o = []
for i in t:
    if ((i < lower_bound) or (i > upper_bound)):
        o.append(i)
# print(f'outliers for the second are {len(o)}')


# checking if there are any empty values or nulls:
X.info()  # No Nulls were found.
y.info()  # No Nulls were found.

# Checking of the presence of duplicates:
duplicates_x = X.duplicated()
num_duplicates = duplicates_x.sum()  # Output: 0
print("number of duplicates in X:", num_duplicates)

duplicates_y = y.duplicated()
num_duplicates1 = duplicates_y.sum()  # Output: 0
print("number of duplicates in Y:", num_duplicates1)

# Getting number of unique values in each column
n = X.nunique(axis=0)
print("No.of.unique values in each column in X:\n", n)

# save values to fill nulls
c = ['Sales', 'Discount']
for i in c:
    mean_value = df[i].mean()
    pickle.dump(mean_value, open(f'{i}.pickle', 'wb'))
    X[i] = X[i].fillna(mean_value)

pickle.dump(df['Quantity'].median(), open('Quantity.pickle', 'wb'))

mode = df[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID', 'CategoryTree',
           'Order Date', 'Ship Date']].mode()

pickle.dump(mode, open(f'mode.pickle', 'wb'))

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

# TIMESERIESANALYSIS Code
matplotlib.rcParams['figure.figsize'] = [25.0, 4.0]  # specify el figure size
firstData = pd.DataFrame({'D': X['Order Date']})
firstData['Profit'] = y
firstData['D'] = pd.to_datetime(firstData['D'])  # step 1
firstData = firstData.sort_values('D')
firstData = firstData.set_index('D')  # step 2
# step 3
start_date = firstData.index.min()
end_date = firstData.index.max()
print(f"start date :{start_date},end date:{end_date}")
firstData = firstData[~firstData.index.duplicated(keep='first')]
# firstData = firstData.reindex(freq) # reindex with freq
# firstData = firstData.dropna() # fill missing values with last observation
# plt.plot(firstData)
plt.title('Time series data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
dec = sm.tsa.seasonal_decompose(firstData, model='additive', freq=12)
dec.plot()
model = ARIMA(firstData, order=(1, 1, 1))
result = model.fit()
print(result.summary())

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
pickle.dump(ordinal, open('Order Month.pickle', 'wb'))

ordinal1 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Order Year'] = ordinal1.fit_transform(np.array(X_train['Order Year']).reshape(-1, 1))
X_test['Order Year'] = ordinal1.transform(np.array(X_test['Order Year']).reshape(-1, 1))
pickle.dump(ordinal1, open('Order Year.pickle', 'wb'))

ordinal2 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=13)
X_train['Ship Month'] = ordinal2.fit_transform(np.array(X_train['Ship Month']).reshape(-1, 1))
X_test['Ship Month'] = ordinal2.transform(np.array(X_test['Ship Month']).reshape(-1, 1))
pickle.dump(ordinal2, open('Ship Month.pickle', 'wb'))

ordinal3 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=6)
X_train['Ship Year'] = ordinal3.fit_transform(np.array(X_train['Ship Year']).reshape(-1, 1))
X_test['Ship Year'] = ordinal3.transform(np.array(X_test['Ship Year']).reshape(-1, 1))
pickle.dump(ordinal3, open('Ship Year.pickle', 'wb'))

ordinal4 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Ship Mode'] = ordinal4.fit_transform(np.array(X_train['Ship Mode']).reshape(-1, 1))
X_test['Ship Mode'] = ordinal4.transform(np.array(X_test['Ship Mode']).reshape(-1, 1))
pickle.dump(ordinal4, open('Ship Mode.pickle', 'wb'))

ordinal5 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=4)
X_train['Segment'] = ordinal5.fit_transform(np.array(X_train['Segment']).reshape(-1, 1))
X_test['Segment'] = ordinal5.transform(np.array(X_test['Segment']).reshape(-1, 1))
pickle.dump(ordinal5, open('Segment.pickle', 'wb'))

ordinal6 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=2)
X_train['Country'] = ordinal6.fit_transform(np.array(X_train['Country']).reshape(-1, 1))
X_test['Country'] = ordinal6.transform(np.array(X_test['Country']).reshape(-1, 1))
pickle.dump(ordinal6, open('Country.pickle', 'wb'))

ordinal7 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=50)
X_train['State'] = ordinal7.fit_transform(np.array(X_train['State']).reshape(-1, 1))
X_test['State'] = ordinal7.transform(np.array(X_test['State']).reshape(-1, 1))
pickle.dump(ordinal7, open('State.pickle', 'wb'))

ordinal8 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=5)
X_train['Region'] = ordinal8.fit_transform(np.array(X_train['Region']).reshape(-1, 1))
X_test['Region'] = ordinal8.transform(np.array(X_test['Region']).reshape(-1, 1))
pickle.dump(ordinal8, open('Region.pickle', 'wb'))

ordinal9 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=4)
X_train['Main Category'] = ordinal9.fit_transform(np.array(X_train['Main Category']).reshape(-1, 1))
X_test['Main Category'] = ordinal9.transform(np.array(X_test['Main Category']).reshape(-1, 1))
pickle.dump(ordinal9, open('Main Category.pickle', 'wb'))

ordinal10 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=18)
X_train['Sub Category'] = ordinal10.fit_transform(np.array(X_train['Sub Category']).reshape(-1, 1))
X_test['Sub Category'] = ordinal10.transform(np.array(X_test['Sub Category']).reshape(-1, 1))
pickle.dump(ordinal10, open('Sub Category.pickle', 'wb'))

ordinal11 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=500)
X_train['City'] = ordinal11.fit_transform(np.array(X_train['City']).reshape(-1, 1))
X_test['City'] = ordinal11.transform(np.array(X_test['City']).reshape(-1, 1))
pickle.dump(ordinal11, open('city.pickle', 'wb'))

ordinal12 = preprocessing.OrdinalEncoder(dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=1541)
X_train['Product ID Number'] = ordinal12.fit_transform(np.array(X_train['Product ID Number']).reshape(-1, 1))
X_test['Product ID Number'] = ordinal12.transform(np.array(X_test['Product ID Number']).reshape(-1, 1))
pickle.dump(ordinal12, open('Product ID Number.pickle', 'wb'))

# Scaling:
# using MinMax
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
pickle.dump(scaler, open('scalerrr.pickle', 'wb'))

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

# Using Anova:
selector = SelectKBest(score_func=f_regression, k=5)
selected_categorical_features = selector.fit_transform(categorical_features, y_train['Profit'])
new_df1 = pd.DataFrame(selected_categorical_features,
                       columns=categorical_features.columns[selector.get_support(indices=True)])

categorical_features["Profit"] = y_train["Profit"].values

# Using Kendall's:
corr = categorical_features.corr(method="kendall")
selected_categorical_features2 = corr.index[abs(corr["Profit"]) > 0]
plt.subplots(figsize=(12, 8))
top_corr1 = categorical_features[selected_categorical_features2].corr()
sns.heatmap(top_corr1, annot=True)
plt.title("Kendall's")
plt.show()

# Numerical Methods:
numerical_features = X_train[['Sales', 'Quantity', 'Discount']]
numerical_features["Profit"] = y_train["Profit"].values

# using spearman's
corr = numerical_features.corr(method="spearman")
selected_num_features2 = corr.index[abs(corr["Profit"]) > 0]
plt.subplots(figsize=(12, 8))
top_corr2 = numerical_features[selected_num_features2].corr()
sns.heatmap(top_corr2, annot=True)
plt.title("Spearman's")
plt.show()

# Using Pearson's:
corr = numerical_features.corr(method="pearson")
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.title("Pearson's")
plt.show()

selected_features = corr.index[abs(corr["Profit"]) > 0.2]
new_df2 = numerical_features[selected_features.delete(-1)]
X_train_new = pd.concat([new_df1, new_df2], axis=1, join="inner")

print("X_train_new-------------------------------")
X_train_new.info()

# feature selection on test set:
cat_features = X_test[
    ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID Number', 'Main Category',
     'Sub Category', 'Order Month', 'Order Year', 'Ship Month', 'Ship Year']]

X_test_cat = cat_features.iloc[:, selector.get_support(indices=True)]
X_test_num = X_test[selected_features.delete(-1)]
X_test_new = pd.concat([X_test_cat, X_test_num], axis=1, join="inner")

print("X_test_new-------------------------------")
X_test_new.info()


# Polynomial regression
def polynomialFeature(degree, x_train=X_train_new, y_tr=y_train, x_test=X_test_new,
                      y_te=y_test):  # train and test data using polynomial regression
    features = PolynomialFeatures(degree)
    # transforms the existing features to higher degree features.
    X_train = features.fit_transform(x_train)
    pickle.dump(features, open(f'FEAT.pickle', 'wb'))
    # fit the transformed features to Linear Regression
    model = linear_model.LinearRegression()
    scores = cross_val_score(model, X_train, y_tr, scoring='neg_mean_squared_error', cv=5)
    model_1_score = abs(scores.mean())
    model.fit(X_train, y_tr)
    # saving the model after traning
    pickle.dump(model, open(f'polynomial of dgree {degree}.pickle', 'wb'))
    print(f"model of degree {degree}'s cross validation score is " + str(model_1_score))
    prediction = model.predict(features.transform(x_test))
    print(f'Model of degree {degree} Test Mean Square Error', metrics.mean_squared_error(y_te, prediction))
    print(f"Accuracy score is {r2_score(y_te, prediction)}")
    plt.scatter(y_te, prediction)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Polynomial Regression')
    plt.show()


def regressionModel(algo, alphaa=1.0):
    if algo == 'linear' or algo == 'Linear':
        model = linear_model.LinearRegression()
    elif algo == 'Lasso' or algo == 'lasso':
        model = Lasso(alpha=alphaa)
    elif algo == 'Ridge' or algo == 'ridge':
        model = Ridge(alpha=alphaa)
    else:
        raise ValueError('Invalid algorithm specified.')
    scores = cross_val_score(model, X_train_new, y_train, scoring='neg_mean_squared_error', cv=5)
    model_1_score = abs(scores.mean())
    # fitting the model with training data
    model.fit(X_train_new, y_train)
    print(f"model of {algo}'s cross validation score is " + str(model_1_score))
    # predicting on test data
    y_pred = model.predict(X_test_new)
    print(f"for {algo} MSE = {metrics.mean_squared_error(y_test, y_pred)}")
    print(f"Accuracy score is {r2_score(y_test, y_pred)}")
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(f'{algo} Regression')
    plt.show()


def Regression(algo, degree=1, alphaa=1.0):
    if algo == 'Polynomial' or algo == 'polynomial' or algo == 'poly':
        polynomialFeature(degree)
    else:
        regressionModel(algo, alphaa)


Regression('polynomial', 2)
Regression('poly', 3)
Regression('linear')
Regression('Ridge', alphaa=-0.9)
Regression('Lasso', alphaa=-0.9)
