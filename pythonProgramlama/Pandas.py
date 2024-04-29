###########################################################
# PANDAS
###########################################################


#########################
# Pandas Series
#########################
# Zamana bağlı serilerde pandas kullanılır

import pandas as pd

# Liste veya farklı tipte verileri pandas serisine çevirir
s = pd.Series([10, 85, 96, 21, 6])
type(s)

s.index
s.dtype
s.size # eleman sayısı
s.ndim # boyut bilgisi (pandas serileri tek boyut)
s.values

type(s.values)  # çıktısı numpy arrays.head(3)
s.head(3)  # baştan ilk üç değer
s.tail(3)  # sondan üç değer

df = pd.read_csv("python_for_data_science/data_analysis_with_python/datasets/advertising.csv")
df.head()

# pandas cheat sheet aramasıyla pythondaki

import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()

#objektler de kategorik değişkendir. sadece bazı fonksiyonlarda farklılık gösterir.

df.columns
df.index
df.describe().T # datestte sayısal olan verilerin analizleri gelir.
df.isnull().values.any()

df.isnull().sum() #her bir değişkende eksik değer hesaplaması

df["sex"].head()

df["sex"].value_counts()


#########################
# Pandas'ta Seçim İşlemleri
#########################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10) # kalıcı olarak silinmedi

# kalıcı olarak silmek için;
# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)


#########################
# Değişkeni indexe çevirmek
#########################

df["age"].head()
df.age.head()
df.index = df["age"]  #indexler değişken oldu

df.drop("age", axis=1).head() #değişiklik sütunda olduğundan axis=1
df.drop("age", axis=1, inplace=True)


#########################
# İndexi değişkene çevirmek
#########################

df.index
df["age"] = df.index
df.head()

df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()


#########################
# Değişkenler üzerinde işlemler
#########################

pd.set_option("display.max_columns", None) #Tüm sütunları gösterir
df.head()

"age" in df # bu değişken df'de var mı?

df["age"].head()
df.age.head()
type(df["age"].head()) # pandas.core.series.Series

# Değişkenin tip bilgisinin dataframe olarak kalmasını istiyorsak çift köşeli parantez kullanmalıyız.
df[["age"]].head()
type(df[["age"]].head()) # pandas.core.frame.DataFrame


df[["age", "alive"]]

col_names = ["age", "alive", "adult_male"]

df[col_names]

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
df.head()

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()


# Belirli bir ifade içeren değişkenleri silmek istediğimizde;
df.loc[:, ~df.columns.str.contains("age")].head() # "~" değildir anlamına gelir.


#########################
# iloc & loc
#########################

# iloc (integer based selection) & loc (label based selection)


import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df.iloc[0:3]    # 3. indexe kadar getirdi
df.iloc[0, 0]

df.loc[0:3]     # 3. index dahil olmak üzere getirdi

df.iloc[0:3, 0:3]   # iloc labelları anlamaz, integer olarak girmeli


df.loc[0:3, "age"] # hata alırız. iloc bizden index bekler
df.loc[0:3, "age"]

col_names = ["age", "alive", "adult_male"]

df.loc[0:3, col_names]


#########################
# Koşullu seçim
#########################

df[df["age"] > 50].head()

df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, "class"].head()

df.loc[df["age"] > 50, ["age", "class"]].head()

# Birden fazla koşul olduğunda koşullar parantez içine alınmalı
# "|" ya da demek

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()


#########################
# Toplulaştırma & Gruplama
#########################

# Toplulaştırma; özet istatistik veren fonskiyonlardır.
# gruopby ile kullanılır (pivot table hariç)
# count()
# first()
# last()
# mean()
# median()
# min()
# max()
# std()
# var()
# sum()
# pivot table

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

# İstenilen grup kırılımında hesaplama yapar (cinsiyete göre yaş)
df.groupby("sex")["age"].mean()

# "agg" fonksiyonuyla birden fazla işlem yaptırılabilir. Daha kullanışlıdır.
df.groupby("sex").agg({"age": ["mean", "sum"]})

# embarke_town'da cinsiyet frekansı verdi
# Sözlük ile kullanmak daha iyi. Fazla işlem yaparken daha yararlı.
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count"})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

# İki seviyeli kırılım

df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean", "sum"],
    "survived": "mean",
    "sex": "count"})


#########################
# Pivot Tablo
#########################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

# values; kesişimlerde ne görülecek değer, index; satırlar; columns; sütunlar
df.pivot_table("survived", "sex", "embarked")

# Kesişimde ortalama var. Pivot tablonun ön tanımlı değeri mean'dir (ortalama).

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

# aggfunc ile standart sapma hesapladık.

df.pivot_table("survived", ["sex", "alive"], ["embarked", "class"], ["count", "mean"])


# cut & qcut fonksiyonları elimizdeki sayısal değişkenleri kategorik değişkene çevirmeye yarar
# cut; elimizdeki değişkenleri hangi aralıkta böleceğimizi biliyorsak.
# qcut; elimizdeki değişkeni hangi aralıkta böleceğimizi bilmiyorsak kendi böler. Küçükten büyüğe sıralar öyle böler

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df["new_age2"] = pd.qcut(df["age"], 4)

df.drop("new_age2", axis= 1)

new_df = df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])


frekans = df["sex"].value_counts()

new_df["frekans"] = frekans


#########################
# Apply & Lambda
#########################
# Kullan at fonksiyonlar

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df.head()

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()


# Klasik Yöntem;

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df.head()

# Lamda ve Apply ile;

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head() #standartlaştırma yaptık

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()


#########################
# Birleştirme
#########################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))

# pd.DataFrame; sıfırdan dataframe oluşturur. ilk argümana girilen değeri (liste olabilir) df'e çevirir.
# 2. argümana değişken isimleri girilir.
df1 = pd.DataFrame(m, columns= ["var1", "var2", "var3"])
df2 = df1 + 99

# Concat
pd.concat([df1, df2]) # indexler tekrarlandı. Düzeltmek için;
pd.concat([df1, df2], ignore_index=True)

pd.concat([df1, df2], axis=1) #yan yana birleştirmek için. axisisn ön değeri 0dır.

# Merge
df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})
df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: her çalışanın müdür bilgisine erişmek
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                    "manager": ["Caner", "Mustafa", "Berkcan"]})

pd.merge(df3, df4)