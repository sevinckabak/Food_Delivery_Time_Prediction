###########################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu - FLO
###########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


#################################
# Görev 1: Veriyi Hazırlama
#################################

#################################
# Adım 1: flo_data_20K.csv verisini okutunuz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
#################################

df = pd.read_csv('Python/Machine_Learning/machine_learning/datasets/flo_data_20k.csv')
df = df.set_index('master_id')
df = df.rename_axis(None)

df.head()

#################################
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#################################

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_cost_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

today_date = df["last_order_date"].max() + pd.Timedelta(days=2)
rfm = pd.DataFrame()

df["recency"] = [(today_date - date).days for date in df["last_order_date"]]

rfm = df[["recency", "total_order", "total_cost_value"]]

rfm.columns = ["recency", "frequency", "monetary"]

#################################
# Görev 2: K-Means ile Müşteri Segmentasyonu
#################################

#################################
# Adım 1: Değişkenleri standartlaştırınız.
#################################

sc = MinMaxScaler(feature_range=(0, 1))
df_sc = sc.fit_transform(rfm)

#################################
# Adım 2: Optimum küme sayısını belirleyiniz.
#################################

kmeans = KMeans()
ssd = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_sc)
    ssd.append(kmeans.inertia_)


plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(1, 10))
elbow.fit(df_sc)
elbow.show()

elbow.elbow_value_

# optimum küme sayısı; 3

#################################
# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz
#################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_sc)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv('Python/Machine_Learning/machine_learning/datasets/flo_data_20k.csv')
df = df.set_index('master_id')
df = df.rename_axis(None)
df["kmeans_cluster_no"] = clusters_kmeans
df.head()

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1

df.describe().T

#################################
# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
#################################


df.groupby("kmeans_cluster_no").agg({""})
df.describe().T

df.to_csv("kmeans_cluster_no.csv")

#################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#################################


#################################
# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
#################################

hc_average = linkage(df, "average")
# öklid uzaklığına göre gözlem birimlerini kümelere ayırır.

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10,)
plt.show()

plt.figure(figsize=(15, 10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

#################################
# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
#################################

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='y', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

#################################
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
#################################


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df= pd.read_csv('Python/Machine_Learning/machine_learning/datasets/USArrests.csv', index_col=0)

df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df.head()

