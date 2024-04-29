#######################################################
# RFM ile Müşteri Segmentasyonu
#######################################################

# 1. İş Problemi
# 2. Veriyi Anlama
# 3. Veri Hazırlama
# 4. RFM Metriklerinin hesaplanması
# 5. RFM Skorlarının Hesaplanması
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi
# 7. Tüm Sürecin Fonskiyonlaştırılması



#####################################
# 1. İş Problemi
#####################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.
# Veri seti gerçektir. Hediyelik eşya satar. Online satış mağazasıdır.
# 1/12/2009 - 09/12/2011 yılları arası satışları kapsar.

# Değişkenler

# InvoiceNo: Fatura Numarası. Her faturaya ait yeni eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her ürün için eşsiz kod.
# Dewscrition: Ürün İsmi
# Quantity: Ürün adedi. Kaç adet ürün satıldığını ifade eder.
# InvoiceDate: Fatura tafihi
# UnitPrice: Ürün fiyatı(Sterlin)
# CustomerID: Eşsiz müşteri numarası.
# Country: Müşterinin yaşadığı ülke ismi.


#####################################
# 2. Veriyi Anlama
#####################################

import datetime as dt
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.reset_option("display.max_rows", False)

df_ = pd.read_excel(r"C:\Users\Sevinç Kabak\PycharmProjects\semih\Python\pythonProgramlama\python_for_data_science\data_analysis_with_python\datasets\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape

df.isnull().sum()
# Customer ID    107927
# müşteri segmentasyonu yapılacağından müşteri bilgisi olmayan gözlemleri değerlendirmeyip sileceğiz.

# eşsiz ürün sayısı?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head()

df["Invoice"].nunique()

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()


#####################################
# 3. Veri Hazırlama
#####################################

df.shape
df.isnull().sum()
df.dropna(inplace=True)

# Invoice'da başında c olan ifadeler iptal faturlardır.

df.describe().T
# iadelerden dolayı fiyatlarda - değerler var bunları uçurmalı.

df = df[~df["Invoice"].str.contains("C", na=False)]
df.head()


#####################################
# 4. RFM Metriklerinin hesaplanması
#####################################

# Recency, Frequency, Monetary
df.head()

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                    "Invoice": lambda Invoice: Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns = ["recency", "frequency", "monetary"]

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]
rfm.shape


#####################################
# 5. RFM Skorlarının Hesaplanması
#####################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
# pd.qcut; veriyi küçükten büyüğe sıralar.
# recency 'de ters oran olduğundan ne kadar küçükse o kadar iyidir.
# Bu sebeple en küçük ilk bölüme 5 değeri verilir.

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Dikkat! frequancy hesaplarken aşağıdaki kodu uyguladığımızda hata alırız.
# Bölünen aralıklar arasına çok fazla aynı değer gelmiş, öyle ki; il aralığa bir değeri gelmişse ikinci aralığa da aynı değer gelmiş.
# Bu noktada rank metodunu kullanmalıyız.
rfm["frequency_score"] = pd.qcut(rfm["frequency"], 5, labels=[1, 2, 3, 4, 5])

# rank ile;
# İlk gördüğü değeri ilk gruba atar.
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])


rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

rfm[rfm["RF_SCORE"] == "55"]

rfm[rfm["RF_SCORE"] == "11"]


#####################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi
#####################################

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
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "cant_loose"].head()

rfm[rfm["segment"] == "new_customers"].index
rfm[rfm["segment"] == "cant_loose"].index

new_df = pd.DataFrame()

new_df["new_customers_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customers_id"] = new_df["new_customers_id"].astype(int)

new_df.to_csv("new_customers.csv")

rfm.to_csv("rf.csv")


#####################################
# 7. Tüm Sürecin Fonskiyonlaştırılması
#####################################

def create_rfm(dataframe, csv=False):

    # Veriyi Hazırlama
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM Metriklerinin Hesaplanması
    today_date = dt.datetime(2010, 12, 11)
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                         "Invoice": lambda Invoice: Invoice.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]

    # RFM Skorlarının Hesaplanması

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # cltc_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                       rfm["frequency_score"].astype(str))

    # Segmentlerin İsimlendirilmesi
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
        r"5[4-5]": "champions"
    }

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]

    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv=True)


# Fonskiyondaki basamaklar ayrı ayrı hesaplanabilir.
# Bu şekilde müdahale işlemi yapılabilir.

# Firmalarda bu analizleri düzenli olarak yapmak gerekli.

