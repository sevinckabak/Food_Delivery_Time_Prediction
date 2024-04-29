
#########################################################
# NUMPY
#########################################################

# Neden Numpy?
# Sabit tipte veri tutar. Verimli veri saklar.
# Hızlı çalışma imkanı sunar.
# Vektörel işlemler (yüksek seviye işlemler)

import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

# normal yaparsak;
ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

# numpy ile yaparsak;
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

a * b

#########################################################
# NumPy Array'i oluşturmak
#########################################################

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype= int)
np.random.randint(0, 10, size=10) # 0-10 aralığında rastgele 10 sayı üretildi

# ortalaması 10 standart sapması 4 olan 3' 4'lük matris oluşturma;
np.random.normal(10, 4, (3, 4))


#########################################################
# NumPy Array Özellikleri
#########################################################

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype


#########################################################
# Yeniden Şekillendirme (Reshaping)
#########################################################

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3,3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3,3)


#########################################################
# Index Seçimi
#########################################################

a = np.random.randint(10, size=10)
a[0]
a[0:5]  #slice
a[0] = 999

m = np.random.randint(10, size=(3, 5))

m[0, 0]  # virgülün sağ tarafı satırları, sol tarafı sütunları temsil eder
m[1, 2]
m[2, 3]
m[2, 3] = 999

m[2, 3] = 2.9 # float ifadeler giremeyiz çünkü numpy sadece belirli bir tip değeri tutar

m[:, 0] # tüm satıların 0. sütunu
m[1,:] # tüm sütunların 1. satırı
m[0:2, 0:3]

#########################################################
# Fancy Index
#########################################################

# arange ifadesi 0'dan 30'a kadar 3'er 3'er artan array oluşturdu;
v = np.arange(0, 30, 3) # 30 hariç
v[1]
v[4]

catch = [1, 2, 3]
# carch içerisindeki sayıları index olarak alır karşılık gelen değeri seçer.
v[catch]


#########################################################
# Numpy'da Koşullu İşlemler
#########################################################

v = np.array([1, 2, 3, 4, 5])

# Klasik Döngü İle

[i for i in v if i < 3]


# Numpy ile

v < 3
v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v <= 3]


#########################################################
# Matematiksel İşlemler
#########################################################

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

########################
# Numpy ile İki Bilinmeyenli Denklem Çözümü
########################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1 ,3]]) # denklemin sağ tarafının katsayıları
b = np.array([12, 10]) # denklemin sol tarafının kat sayıları

np.linalg.solve(a, b)

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('1. boyuttaki 2.eleman: ', arr[0, 1])

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])