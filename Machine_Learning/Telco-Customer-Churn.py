################################################
# Telco-Customer-Churn
################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows', None)

# Görev 1: Keşifçi Veri Analizi

df = pd.read_csv("Python/Machine_Learning/machine_learning/datasets/Telco-Customer-Churn.csv")

def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım 1. Nümerik ve Kategorik Değişken Analizi

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.describe().T

df.head()
df["TotalCharges"].nunique()
df["TotalCharges"].isnull().sum()

num_cols.append('TotalCharges')

df.info()

# Adım 2. Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# TotalCharges değişkenini floata çevirelim.
df["TotalCharges"] = df["TotalCharges"].apply(lambda x: x == 0 if x == " " else x)

bosluk = df.loc[df["TotalCharges"] == " ", "TotalCharges"].index

df["TotalCharges"] = df["TotalCharges"].astype(float)

# Ek çözüm
df["TotalCharges"].replace(" ", np.nan, inplace=True)
df["TotalCharges"].fillna(0)

df.info()

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def num_summary(dataframe, numerical_col, plot=False):
    quatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quatiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, num_cols, True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, True)

for col in cat_cols:
    print(df.groupby([col, 'Churn']).count())

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Churn", cat_cols)


# Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))
# aykırı gözlem yok.


# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()

# TotalCharges değişkeni 0 olamaz çünkü hiçbir değişkenin MonthlyCharges değeri 0 değil. Bu sebeple 0 olan değişkenleri nan yaptım.
df.loc[df["TotalCharges"] == 0, "TotalCharges"]
df["TotalCharges"] = df["TotalCharges"].apply(lambda x: np.nan if x == 0 else x)

# cat_cols değişkenlerinde boş değer olup olmadığını kontrol ettim.
for col in cat_cols:
    print(col, df.loc[df[col] == " ", col])

df.shape


# Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# TotalCharges değişkenindeki boş değerler kategorik kırılımda ortalamayla dolduruldu.

df.groupby("Contract")["TotalCharges"].mean()

df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("Contract")["TotalCharges"].transform("mean"))

df.isnull().sum()


# Adım 2: Yeni değişkenler oluşturunuz.

df.head()
df.drop(["customerID"], inplace=True, axis=1)
df.drop(["tenure_cat"], inplace=True, axis=1)

# Tenure değişkenini kategorik hale getirme
df["new_tenure_cat"] = pd.qcut(df["tenure"], q=3, labels=["short_term", "mid_term", "long_term"])

# Tenure değişkenine göre internet servisleri
df.loc[(df['InternetService'] == 'DSL') & (df['new_tenure_cat'] == "short_term" ), 'new_int_tenure'] = 'short_term_DSL'
df.loc[(df['InternetService'] == 'DSL') & (df['new_tenure_cat'] == "mid_term" ), 'new_int_tenure'] = 'mid_term_DSL'
df.loc[(df['InternetService'] == 'DSL') & (df['new_tenure_cat'] == "long_term" ), 'new_int_tenure'] = 'long_term_DSL'
df.loc[(df['InternetService'] == 'Fiber optic') & (df['new_tenure_cat'] == "short_term" ), 'new_int_tenure'] = 'short_term_Fiber optic'
df.loc[(df['InternetService'] == 'Fiber optic') & (df['new_tenure_cat'] == "mid_term" ), 'new_int_tenure'] = 'mid_term_Fiber optic'
df.loc[(df['InternetService'] == 'Fiber optic') & (df['new_tenure_cat'] == "long_term" ), 'new_int_tenure'] = 'long_term_Fiber optic'
df.loc[(df['InternetService'] == 'No') & (df['new_tenure_cat'] == "short_term" ), 'new_int_tenure'] = 'short_term_No int'
df.loc[(df['InternetService'] == 'No') & (df['new_tenure_cat'] == "mid_term" ), 'new_int_tenure'] = 'mid_term_No int'
df.loc[(df['InternetService'] == 'No') & (df['new_tenure_cat'] == "long_term" ), 'new_int_tenure'] = 'long_term_No int'

# müşterinin yalnız olma durumu
df.loc[((df['Partner'] + df['Dependents']) > 0), "new_is_alone"] = "NO"
df.loc[((df['Partner'] + df['Dependents']) == 0), "new_is_alone"] = "YES"

# İnternet ve telefon servisleri
df.loc[(df["PhoneService"] == 1) & (df["InternetService"] != "No"), "new_services"] = "phone & internet"

df.loc[(df["PhoneService"] == 0) & (df["InternetService"] != "No"), "new_services"] = "internet"

df.loc[(df["PhoneService"] == 1) & (df["InternetService"] == "No"), "new_services"] = "phone"


# Diğer Değişkenler

contract_duration = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractDuration'] = df['Contract'].map(contract_duration)

df['ElectronicPayment'] = df['PaymentMethod'].apply(lambda x: 1 if 'electronic' in x.lower() else 0)

df['LoyaltyScore'] = df['tenure'].apply(lambda x: x // 12)  # Her 12 ay için 1 puan

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SpendingRatio'] = df['MonthlyCharges'] / df['TotalCharges']
df['SpendingRatio'] = df['SpendingRatio'].replace([np.inf, -np.inf], np.nan)  # Sıfıra bölme hatalarını NaN ile değiştir

df['SeniorTechUse'] = ((df['SeniorCitizen'] == 1) &
                             (df['InternetService'] != 'No')).astype(int)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """

    Veri setindeki kategorik değşkenler için one hot encoding işlemini yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder
    categorical_cols : Kategorik değişkenleri ifade eder
    drop_first : Dummy değişken tuzağına düşmemek için ilk değşşkeni siler

    Returns
    -------
    One-hot encoding işlemi yapılmış bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "pandas" kütüphanesine bağımlılığı bulunmaktadır.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype='int')
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]


df.head()
df = label_encoder(df, "new_is_alone")

def rare_encoder(dataframe, rare_perc):
    """

    Verilen veri setinde, önceden verilen orana göre rare encoding işlemi yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder.
    rare_perc : Nadir görülme oranını ifade eder.

    Returns
    -------
    Rare encoding yapılmış datafremi return eder
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()
df.drop("customerID", inplace=True, axis=1)
df.head()
df.shape

# Görev 3 : Modelleme

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

# Tüm veriyi kullanarak model kuruyoruz.

y = df["Churn"]

X = df.drop(["Churn"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))

# accuracy = 0.81
# precision = 0.68
# recall = 0.54
# f1-score = 0.60


# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.


# Hold out sistemi ile verinin bir kısmını bölerek model kuruldu.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

# accuracy = 0.80
# precision = 0.68
# recall = 0.51
# f1-score = 0.58

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
# 0.83

# 5 Katlı Cross-Validation yöntemi ile tahmin
log_model = LogisticRegression().fit(X, y)

cv_result = cross_validate(log_model,
                           X, y,
                           cv=5,
                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_result['test_accuracy'].mean()
# Accuracy: 0.8044

cv_result['test_precision'].mean()
# Precision: 00.6647

cv_result['test_recall'].mean()
# Recall: 0.5313

cv_result['test_f1'].mean()
# F1-score: 0.5905

cv_result['test_roc_auc'].mean()
# AUC: 0.0.8457


# KNN Model

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier().fit(X, y)

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)


# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.84
# f1 0.67

# AUC
roc_auc_score(y, y_prob)
# 0.89

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.77
cv_results['test_f1'].mean()
# 0.55
cv_results['test_roc_auc'].mean()
# 0.77

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

from sklearn.model_selection import GridSearchCV
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8026
cv_results['test_f1'].mean()
# 0.5908
cv_results['test_roc_auc'].mean()
# 0.8344


# Farklı bir kütüphane ile model kurulumu
import statsmodels.api as sm

X = df[["TV"]]
X = sm.add_constant(X)
X[0:5]
y = df[["sales"]]
lm = sm.OLS(y, X)
model = lm.fit()
model.summary()
model.summary().tables[1]
model.conf_int()
model.params
model.f_pvalue
print("f_pvalue: ", "%.4f" % model.f_pvalue)
print("fvalue: ", "%.3f" % model.fvalue)
print("tvalue: ", "%.2f" % model.values[0:1])
model.mse_model
model.rsquared
model.rsquared_adj
model.fittedvalues[0:5]  # tahmini y değerleri
print("Sales= " + str("%.2f" % model.params[0]) + " + TV" + "*" + "%.2f" % model.params[1])