import pickle
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import preprocessing, linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('megastore-tas-test-regression.csv')

X = df.loc[:, df.columns != 'Profit']

y = df[['Profit']]

def extract_features(string):
    split_string = string.split('-')
    feature3 = split_string[2]
    return feature3


def prepreprocess(X):
    c = ['Sales', 'Discount']
    for i in c:
        mean_value = pickle.load(open(f'{i}.pickle', 'rb'))
        X[i] = X[i].fillna(mean_value)

    Quantity_median = pickle.load(open('Quantity.pickle', 'rb'))
    X['Quantity'] = X['Quantity'].fillna(Quantity_median)

    mode = pickle.load(open(f'mode.pickle', 'rb'))
    X[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID', 'CategoryTree',
       'Order Date', 'Ship Date']] = X[
        ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Product ID', 'CategoryTree',
         'Order Date', 'Ship Date']].fillna(mode)

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
    MainSub = pd.json_normalize(X['CategoryTree'].apply(eval))
    X['Main Category'] = MainSub['MainCategory']
    X['Sub Category'] = MainSub['SubCategory']

    X.drop('CategoryTree', axis=1, inplace=True)

    X['Product ID Number'] = X['Product ID'].apply(extract_features)
    X.drop('Product ID', axis=1, inplace=True)

    return X


X_test_sample = prepreprocess(X)

for col in ['Order Month', 'Order Year', 'Ship Month', 'Ship Year', 'Ship Mode', 'Segment', 'Country', 'State',
            'Region', 'Main Category', 'Sub Category', 'City', 'Product ID Number']:
    enc = pickle.load(open(f'{col}.pickle', 'rb'))
    X_test_sample[col] = enc.transform(np.array(X_test_sample[col]).reshape(-1, 1))


scaler = pickle.load(open('scalerrr.pickle', 'rb'))
X_test_sample = pd.DataFrame(scaler.transform(X_test_sample), columns=X_test_sample.columns)

print(X_test_sample)

selected_features = ['City', 'State', 'Region', 'Main Category', 'Sub Category', 'Sales', 'Discount']

X_new = X_test_sample[selected_features]

print(X_new)

features = pickle.load(open(f'FEAT.pickle', 'rb'))
s_m = pickle.load(open(f'polynomial of dgree 3.pickle', 'rb'))
#features = PolynomialFeatures(3)
prediction = s_m.predict(features.transform(X_new))
print(f"Accuracy score is {r2_score(y, prediction)}")