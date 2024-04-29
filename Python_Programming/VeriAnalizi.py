#########################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ
#########################################################

############################################
# 1 - Genel Resim
############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

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

df2 = sns.load_dataset("tips")
df3 = sns.load_dataset("flights")

check_df(df3)

############################################
# 2- Kategorik Değişken Analizi
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()
df.info()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["bool", "object", "category"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["float64", "int64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in df.columns if col in cat_cols]

df[cat_cols].nunique()


# Fonksiyonda yapılacak işlemler;
df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot= True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        print("Bool tipi olduğundan çalışmaz")
    else:
    cat_summary(df, col, plot= True)

# plot fonksiyonu bool tipini kapsamadığından hata alırız.


df["adult_male"].astype(int)
# tip bool ise tipini int olarak değiştirip o şekilde grafikleştirdik.
for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
    cat_summary(df, col, plot= True)


cat_summary(df, "sex")


############################################
# 3- Sayısal Değişken Analizi
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["bool", "object", "category"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["float64", "int64"]]
cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["bool", "object", "category"] and df[col].nunique() > 10]
cat_cols = cat_cols + num_but_cat

num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col):
    quatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quatiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quatiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)



############################################
# 4- Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# Docstring; fonksiyona argüman yazma

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimleri verilir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        nümerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişkenler listesi
    num_cols: list
        Nümerik değişkenler listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat, cat_cols'un içerisindedir.

    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["bool", "object", "category"]]
    num_but_cat = [col for col in df.columns if
                   df[col].nunique() < 10 and df[col].dtypes in ["float64", "int64"]]
    cat_but_car = [col for col in df.columns if
                   str(df[col].dtypes) in ["object", "category"] and df[col].nunique() > 20]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

help(grab_col_names)

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

cat_summary(df, "survived")

def num_summary(dataframe, numerical_col):
    quatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quatiles).T)

for col in num_cols:
    num_summary(df, col)

df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes =="bool":
        df[col] = df[col].astype(int)

for col in cat_cols:
    cat_summary(df, col, plot=True)



############################################
# 5- Hedef Değişken Analizi
############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes =="bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal değişkenlerin isimleri verilir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        nümerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişkenler listesi
    num_cols: list
        Nümerik değişkenler listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat, cat_cols'un içerisindedir.

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["bool", "object", "category"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["float64", "int64"]]
    cat_but_car = [col for col in dataframe.columns if
                   str(dataframe[col].dtypes) in ["object", "category"] and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show(block=True)

df["survived"].value_counts()
cat_summary(df, "survived")


############################################
# Hedef Değişkenin Kategorik Değişkenlerle Analizi
############################################

df.groupby("adult_male")["survived"].mean()

def target_sum(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(categorical_col)[target].mean()}))


target_sum(df, "survived", "pclass")

for col in cat_cols:
    target_sum(df, "survived", col)


############################################
# Hedef Değişkenin Sayısal Değikenlerle Analizi
############################################

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age": "mean"})


def target_sum_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_sum_num(df, "survived", "age")


for col in num_cols:
    target_sum_num(df, "survived", col)

############################################
# 6- Korelasyon Analizi
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_csv("python_for_data_science/data_analysis_with_python/datasets/breast_cancer.csv")
df.head()

df = df.iloc[:, 1:-1] #istenmeyen baştaki ve sondaki iki değişkenden kurtulmak için yazıldı.
df.head()

num_cols= [col for col in df.columns if df[col].dtype in [int, float]]

# Korelasyon analizi "corr" fonksiyonu ile yapılır.
corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Yüksek kolerasyonlu değişkenlerin silinmesi

cor_matrix = corr.abs()

# Mutlak değer içine aldık

upper_triangle_matris = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))

drop_list = [col for col in upper_triangle_matris.columns if any(upper_triangle_matris[col] > 0.90)]

cor_matrix[drop_list]

df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in [int, float]]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matris = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matris.columns if any(upper_triangle_matris[col].astype(float) > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)
