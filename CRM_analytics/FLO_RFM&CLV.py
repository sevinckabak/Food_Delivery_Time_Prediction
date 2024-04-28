#######################################################
# FLO RFM Analizi İle Müşteri Segmentasyonu
#######################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_csv(
    r"/Python/pythonProgramlama/python_for_data_science/data_analysis_with_python/datasets/flo_data_20k.csv")
df = df_.copy()

# 2. Veri setinde
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boş değer,
df.isnull().sum()
# e. Değişken tipleri, incelemesi yapınız.
df.shape
df.dtypes
# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_cost_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": "count",
                                 "total_order": ['mean', 'sum', 'min', 'max'],
                                 "total_cost_value": ['mean', 'sum', 'min', 'max']})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values(by="total_cost_value", ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values(by="total_order", ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(dataframe):
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_cost_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])
    return dataframe

# GÖREV 2: RFM Metriklerinin Hesaplanması

df.info()

today_date= df["last_order_date"].max() + pd.Timedelta(days=2)

rfm = pd.DataFrame()

df["recency"] = [(today_date - date).days for date in df["last_order_date"]]

# Diğer çözüm
rfm["recency"] = (today_date - df["last_order_date"]).astype("timedelta64[D]")

rfm = df[["master_id", "recency", "total_order", "total_cost_value"]]

rfm.columns = ["master_id", "recency", "frequency", "monetary"]

rfm.describe().T

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

rfm["rfm_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str)

rfm.sort_values(by="frequency").head(30)

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"}

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex=True)


# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.


rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})


# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

new_df = pd.DataFrame()
new_df = pd.merge(rfm, df, on="master_id", how="left")

women_df = new_df.loc[((new_df["segment"] == "loyal_customers") | (new_df["segment"] == "champions"))
                      & (new_df["interested_in_categories_12"].str.contains("KADIN")), ["segment", "interested_in_categories_12", "monetary"]]

women_df.to_csv("women.csv")

# Diğer Çözüm;

target_segment_cust_id = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["master_id"]

cust_ids = df[(df["master_id"].isin(target_segment_cust_id))
              & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

cust_ids.to_csv("cust_ids", index=False)

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

target_segment_cust_id = rfm[(rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"]))]["master_id"]


##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_csv(
    r"/Python/pythonProgramlama/python_for_data_science/data_analysis_with_python/datasets/flo_data_20k.csv")
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)

# replace_with_thresholds; aykırı değerleri limitlerle değiştirir
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.info()

for col in df.columns:
        if df[col].dtypes == "float64":
            replace_with_thresholds(df, col)

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_cost_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df.info()

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

today_date = df["last_order_date"].max() + pd.Timedelta(days=2)

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

cltv_df = pd.DataFrame
df["recency"] = df["last_order_date"] - df["first_order_date"]
df["recency"] = df["recency"].apply(lambda x: x.days / 7)

df["T_weekly"] = today_date - df["first_order_date"]
df["T_weekly"] = df["T_weekly"].apply(lambda x: x.days / 7)

df["monetary"] = df["total_cost_value"] / df["total_order"]

cltv_df = df.loc[:, ["master_id", "recency", "total_order", "T_weekly", "monetary"]]
cltv_df["frequency"] = cltv_df["total_order"]
cltv_df.drop(columns=["total_order"], inplace=True)

cltv_df["recency"] = cltv_df["recency"].astype("int64") / 7
cltv_df.info()


# Diğer Çözüm;
cltv_df = pd.DataFrame

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency_weekly"] = cltv_df[""]



# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T_weekly'])


cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False)

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False)

cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10)

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["exp_avrg_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])







