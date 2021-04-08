import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


rand_dataset = pd.read_csv("sample_submission.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_sales_column = pd.DataFrame(rand_dataset.iloc[: , 1])

# Here test dataset deprives of the price columns which is present in the rand_dataset
# so adding the sales column of rand_dataset as the last column of test dataset
test = pd.concat([test , test_sales_column] , axis = 1)

# Now concatenating those two to make the large dataset
dataset = pd.concat([train , test])

# Removing some features that contains way more null values
to_remove_indices = []
for i in dataset.columns:
    if len(dataset[i].isnull().value_counts()) == 2:
        if(dataset[i].isnull().value_counts()[True] >875):
            to_remove_indices.append(i)
            
dataset = dataset.drop(to_remove_indices , axis = 1)

# Managing the missing values

null_list = []
for i in range(len(dataset.columns)):
    if(len(dataset.iloc[:,i].isnull().value_counts().index) ==2):
        null_list.append(i)
        
categorical_null_list = []
numerical_null_list = []


for i in null_list:
    if(dataset.iloc[: , i].dtype == 'O'):
        categorical_null_list.append(i)
    elif(dataset.iloc[: , i].dtype == "int64" or dataset.iloc[: , i].dtype=="float64"):
        numerical_null_list.append(i)
        
numerical_simple_imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
categorical_simple_imputer = SimpleImputer(missing_values = np.nan , strategy = "most_frequent")

dataset.iloc[: , numerical_null_list] = numerical_simple_imputer.fit_transform(dataset.iloc[: , numerical_null_list])
dataset.iloc[: , categorical_null_list] = categorical_simple_imputer.fit_transform(dataset.iloc[: , categorical_null_list])

# Encoding the categorical data
categorical_indices = []
numerical_indices = []
for i in range(len(dataset.columns)):
    if(dataset.iloc[: , i].dtype=="O"):
        categorical_indices.append(i)
    elif(dataset.iloc[: , i].dtype=="int64" or dataset.iloc[: , i].dtype=="float64"):
        numerical_indices.append(i)
        
categorical_label_indices = []
categorical_onehot_indices = []

for i in categorical_indices:
    if(len(dataset.iloc[: , i].value_counts()) == 2):
        categorical_label_indices.append(i)
    elif(len(dataset.iloc[: , i].value_counts()) > 2):
        categorical_onehot_indices.append(i)
        
        
label_encoder = LabelEncoder()
for i in categorical_label_indices:
    dataset.iloc[: , i] = label_encoder.fit_transform(dataset.iloc[: , i])

# To know the places from where dummy variable is to be removed after oneHotEncoding
b = 0;
i=0;
dummy_var_indices = [];
for j in categorical_onehot_indices:
    a = i + b;
    dummy_var_indices.append(a)
    b = len(dataset.iloc[: , j].value_counts())
    i = a;
    
y = dataset.iloc[: , -1].values

    
ct = ColumnTransformer([("encoder" , OneHotEncoder() , categorical_onehot_indices)] , remainder="passthrough")
X = ct.fit_transform(dataset)

include_var_indices = []
# X.shape[1] means columns in encoded data
for i in range(X.shape[1]):
    if(i not in dummy_var_indices):
        include_var_indices.append(i)
        
X = X[: , include_var_indices]
X = X.toarray()
X = X[: , 0:-1]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Let's try different regression algorithm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lr_score = cross_val_score(LinearRegression() , X , y , cv=10)
svr_score = cross_val_score(SVR() , X , y , cv=10)
dt_score = cross_val_score(DecisionTreeRegressor(random_state = 0) , X , y , cv=10)
rf_score = cross_val_score(RandomForestRegressor(n_estimators = 100) , X , y , cv=10)

# from cross_val_score --> random forest seems to shine
# Now performing feature extraction
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X = pca.fit_transform(X)
explained_variances = pca.explained_variance_ratio_
# extracting features is not favorabel , as every feature have relative variance meaning
# i.e. no 2 or 3 distinct features have dominant variance

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

# Finding the salesprice of houses
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train , y_train)
y_pred = regressor.predict(X_test)





    


    


