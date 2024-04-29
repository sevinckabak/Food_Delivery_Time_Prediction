######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("Python/Machine_Learning/machine_learning/datasets/advertising.csv")


x = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(x,y)

# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?


reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

# Modelin Görselleştirilmesi
g = sns.regplot(x=x, y=y, scatter_kws={"color": "b", "s":9},
                ci=False, color="r")

# ci: güven aralığı bilgisi
# color: regresyon çizgisi rengi
# scatter_kws'deki color; saçılım grafiği (gerçek değerler görselleri)

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
# virgülden sonra iki basamak olacak şekilde yuvarla

g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
# y ekseni sıfırdan başlayacak
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(x)
#regreseyon modeli kullanılarak bağımsız değişkenler verilir ve bağımlı değişkenleri tahmin eder.

mean_squared_error(y, y_pred)
# 10.51

y.mean()
y.std()
# gerçek değerlerim ortalama 14 birim ve standart sapması 5 birim.
# dolayısıyla mse değerim (10 br) oldukça yüksek

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(x, y) #  0.61
# veri setinde bağımsız değişkenlerin bağımlı değişkendeki değişkliği açıklama yüzdesidir.
# Bu örnekte bağımsız değişken bağımlı değişkenin değişikliğini %61 oranında açıklayabilmiştir.
# bağımsız değişkenler arttıkça r-kare şişmeye meyillidir. Düzeltilmiş r-kare değerini göz önünde bulundurmalı.


######################################################
# Multiple Linear Regression
######################################################

x = df.drop("sales", axis=1)
y = df[["sales"]]

##########################
# Model
##########################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

x_train.shape
y_train.shape
y_test.shape

reg = LinearRegression()
reg.fit(x_train, y_train)

reg_model = LinearRegression().fit(x_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

2.90794702 + 0.0468431*30 + 0.17854434*10 + 0.00258619*40

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)


##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(x_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE

reg_model.score(x_train, y_train)
# 0.89
# yeni değişken eklediğimizde başarı arttı hata düştü

# Test RMSE
y_pred = reg_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# genelde test setinin hatası train setinden daha yüksek çıkar

# Test RKARE
reg_model.score(x_test, y_test)


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 x,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 1.69
# scoring; negatif hesaplar bu sebeple eksiyle çarparız.
# veri seti küçük olduğundan tamamını modele verdik.
# veri seti küçük olduğundan 10 katlı çapraz kontrole daha çok güvenebiliriz.


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 x,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w
# tek bir iterasyon yapar.
# verilen b ve w değerlerini tüm gözlemlerden geçirerek ortalamasını alır ve yeni değerleri hesaplar.

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("Python/Machine_Learning/machine_learning/datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# parametre; veri setinden bulunur.
# hipermarametre; kullanıcı tarafından belirlenir.

# hyperparameters (örnek olarak verildi)
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
#After 100000 iterations b = 9.311638095155203, w = 0.2024957833925339, mse = 18.09239774512544

np.sqrt(18.09239774512544)


























