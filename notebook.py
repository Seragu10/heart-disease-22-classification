# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
data =pd.read_csv('/workspaces/heart-disease-22-classification/heart_2022_no_nans.csv')
data.head()


# %%
data.info()

# %%
data.isnull().sum()

# %%
col_name = list(data.dtypes[data.dtypes == 'object'].index)

# %%
col_name

# %%
for c in col_name:
    data[c] = data[c].str.replace(' ', '_').str.lower()

# %%
data.head()

# %%
data.info()

# %%
df = data.copy()

# %%
#columns that can label encoded
label_encoded_col = []

# %%
for col in data.columns:
    if(len(data[col].value_counts().head(11)) < 11):
        label_encoded_col.append(col)

# %%
label_encoded_col

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le  = LabelEncoder()
for col in label_encoded_col:
    data[col] = le.fit_transform(data[col])


# %%
data.head()

# %%
data['GeneralHealth'].value_counts()

# %%
data.info()

# %%
data['HadHeartAttack'].value_counts()

# %%
data = data.drop(columns = 'State')


# %%
data.info()

# %%
from sklearn.metrics import mutual_info_score

# %%
def cal_mutual_info_score(series):
    return mutual_info_score(series, data['HadHeartAttack'])

# %%
categorical = [col for col in data.columns if data[col].dtype == 'object']

# %%
categorical

# %%
data[categorical].value_counts()

# %%
data['AgeCategory'] = le.fit_transform(data['AgeCategory'])

# %%
data.head()

# %%
data.corr()

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = 'HadHeartAttack'), data['HadHeartAttack'], test_size = 0.2, random_state = 42)

# %%
from sklearn.feature_extraction import DictVectorizer

# %%
dv = DictVectorizer(sparse=False)

train_dict = X_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

test_dict = X_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

# %%
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)

# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
from sklearn.metrics import f1_score,accuracy_score

# %%
print(f1_score(y_test, y_pred, average = 'weighted'))

# %%
print(accuracy_score(y_test, y_pred))

# %%
model.score(X_test, y_test)

# %%



