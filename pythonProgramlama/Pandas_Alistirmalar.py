####################################################
# PANDAS ALIŞTIRMALAR
####################################################

#############
# Görev 1
# Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#############
import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df.shape
# çıktı; (891, 15) 891 gözlem, 15 değişken

df.info()

#############
# Görev 2
# Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#############

df["sex"].value_counts()

#############
# Görev 3
# Her bir sutuna ait unique değerlerin sayısını bulunuz.
#############

df.nunique()

#############
# Görev 4
# pclass değişkeninin unique değerlerinin sayısını bulunuz.
#############

df["pclass"].nunique()

#############
# Görev 5
# pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#############

df[["pclass", "parch"]].nunique()

#############
# Görev 6
# embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
#############

df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
# çıktı; CategoricalDtype(categories=['C', 'Q', 'S'], ordered=False) ordered false demesi sıralama gözükmüyor demek.
df["embarked"] = pd.Categorical(df["embarked"])
df["embarked"].dtypes

#############
# Görev 7
# embarked değeri C olanların tüm bilgelerini gösteriniz.
#############

df[df["embarked"] == "C"].head()

#############
# Görev 8
# embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#############

df[df["embarked"] != "S"].head()

#############
# Görev 9
# Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz
#############

df[(df["age"] < 30) & (df["sex"] == "female")].head()

#############
# Görev 10
# Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz
#############

df[(df["fare"] > 500) | (df["age"] > 70)].head()

#############
# Görev 11
# Her bir değişkendeki boş değerlerin toplamını bulunuz
#############

df.isnull().sum()

#############
# Görev 12
# who değişkenini dataframe’den çıkarınız.
#############

df.drop("who", axis=1)

#############
# Görev 13
# deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#############

mode = df["deck"].mode()

df.loc[df["deck"].isnull(), "deck"] = "C"

df["deck"].value_counts()

# diğer çözüm
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()
#############
# Görev 14
# age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz
#############

median = df["age"].median()

df.loc[df["age"].isnull(), "age"] = median

df["age"].value_counts()

# diğer çözüm
df["age"].fillna(df["age"].median())

#############
# Görev 15
# survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz
#############

df.groupby(["sex", "pclass"]).agg({"survived": ["sum", "count", "mean"]})

#############
# Görev 16
# 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
# setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)
#############

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
df.head()

#############
# Görev 17
# Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#############

df_new = sns.load_dataset("tips")

#############
# Görev 18
# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#############

df_new.groupby("time").agg({"total_bill": ["min", "max", "mean"]})

#############
# Görev 19
# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#############

df_new.groupby(["day", "time"]).agg({"total_bill": ["min", "max", "mean", "sum"]})

df_new.pivot_table("total_bill", "day", "time", aggfunc=["min", "max", "mean", "sum"])

#############
# Görev 20
# Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#############

df_female_lunch = df_new[(df_new["time"] == "Lunch") & (df_new["sex"] == "Female")]

df_female_lunch.groupby(["day"]).agg({"total_bill": ["sum", "min", "max", "mean"],
                                      "tip": ["sum", "min", "max", "mean"]})

# Tek işlemde;
df_new[(df_new["time"] == "Lunch") & (df_new["sex"] == "Female")].groupby(["day"]).agg({"total_bill": ["sum", "min", "max", "mean"],
                                      "tip": ["sum", "min", "max", "mean"]})


#############
# Görev 21
# size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
#############

df_new.loc[(df_new["size"] < 3) & (df_new["total_bill"] > 10), ["total_bill"]].mean()

#############
# Görev 22
# total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin
#############

df_new["total_bill_tip_sum"] = df_new["total_bill"] + df_new["tip"]

#############
# Görev 23
# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#############

df_new.sort_values(by="total_bill_tip_sum", ignore_index=True, ascending=False, inplace=True)

df_new_first_thirty = df_new.iloc[0:31, :]

# Çağla Hoca'nın çözümü

new_df = df_new.sort_values(by="total_bill_tip_sum", ascending=False)[:30]