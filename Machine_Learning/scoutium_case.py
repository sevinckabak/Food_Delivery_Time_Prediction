################################################################################################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma - Scoutium Case Study
################################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



#######################################
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
#######################################

attributes = pd.read_excel("Python/Machine_Learning/machine_learning/datasets/Scotium/scoutium_attributes.xlsx")
potential_labels = pd.read_excel("Python/Machine_Learning/machine_learning/datasets/Scotium/scoutium_potential_labels.xlsx")


#######################################
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
#######################################

df = pd.merge(potential_labels, attributes, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how='inner')


#######################################
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
#######################################
def cat_summary(dataframe, col_name, plot=False):
    """

    Fonksiyon, veri setinde yer alan kategorik, numerik vs... şeklinde gruplandırılan değişkenler için özet bir çıktı
    sunar.

    Parameters
    ----------
    dataframe : Veri setini ifade
    col_name : Değişken grubunu ifade eder
    plot : Çıktı olarak bir grafik istenip, istenmediğini ifade eder, defaul olarak "False" gelir

    Returns
    -------
    Herhangi bir değer return etmez

    Notes
    -------
    Fonksiyonun pandas, seaborn ve matplotlib kütüphanelerine bağımlılığı vardır.

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
df.describe().T
df.info()
df.isnull().sum()

cat_summary(df, "position_id")

df = df.drop(df[df['position_id'] == 1].index)
# 700 gözlem silindi.


#######################################
# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
#######################################

cat_summary(df, "potential_label")

df = df.drop(df[df['potential_label'] == "below_average"].index)
# 136 gözlem silindi


#######################################
# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
#######################################

#######################################
# Adım 5.1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
#######################################

df_pivot = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")

#######################################
# Adım 5.2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz
#######################################

df_pivot.columns = df_pivot.columns.astype(str)
df_pivot.reset_index(inplace=True)

#######################################
# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
#######################################

def label_encoder(dataframe, binary_col):
    """
    Fonksiyon verilen veri setindeki ilgili değişkenleri label encoding sürecine tabii tutar.

    Parameters
    ----------
    dataframe: Veri setini ifade eder.
    binary_col: Encode edilecek olaran değişkenleri ifade eder

    Returns
    -------
    Encoding işlemi yapılmiş bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "from sklearn.preprocessing import LabelEncoder" paketine bağımlılığı bulunmaktadır.

    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df_pivot, "potential_label")
# 0: average, 1:highlighted

#######################################
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
#######################################

dff = df_pivot.select_dtypes("float")

#######################################
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
#######################################

dff = pd.DataFrame(StandardScaler().fit_transform(dff), columns=dff.columns)

#######################################
# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
#######################################

X = dff
y = df_pivot["potential_label"]

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
cv_results['test_accuracy'].mean()
# 0.8670634920634921
cv_results['test_precision'].mean()
#  0.85
cv_results['test_recall'].mean()
#  0.4699999999999999
cv_results['test_f1'].mean()
# 0.5851587301587301
cv_results['test_roc_auc'].mean()
# 0.9024603174603175

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

# Mehmet'in çözümü

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('GBM', best_models["GBM"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
cv_results['test_accuracy'].mean()
# 0.8783068783068784
cv_results['test_precision'].mean()
#  1.0
cv_results['test_recall'].mean()
#  0.42000000000000004
cv_results['test_f1'].mean()
# 0.5547979797979797
cv_results['test_roc_auc'].mean()
# 0.9025324675324675

#######################################
# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X, num=5)

















