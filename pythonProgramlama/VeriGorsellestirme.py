######################################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIN & SEABORN
######################################################

######################################################
# MATPLOTLIB
######################################################

# Veri görselleştirme tekniklerinin atası
# Low level

######################################################
# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot
import matplotlib.pyplot as plt
import numpy as np
# Kategorik değişken: sütun grafik. countplot (matplotlib) / barplot (seaborn)
# Sayısal değişken: hist, boxplot

############################################
# Kategorik değişken veri görselleştirme
############################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind= "bar")
plt.show()


############################################
# Sayısal değişken veri görselleştirme
############################################

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])


############################################
# Matplotlib Özellikleri
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

######################
# plot
######################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

######################
# marker
######################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker = "o")
plt.show()

markers = ["o", "*", ".", ",", "x", "X", "+", "P", "p", "s", "d", "D", "H", "h"]


######################
# Line
######################

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle = "dashed")
plt.plot(y, linestyle = "dotted")
plt.plot(y, linestyle = "dashdot", color = "r")


######################
# Multiple Lines
######################

 x = np.array([23, 18, 31, 10])
 y = np.array([13, 28, 11 , 100])
 plt.plot(x)
 plt.plot(y)


######################
 # Labels
######################

 x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
 y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
 plt.plot(x,y)

# Başlık
 plt.title("Ana Başlık")
# x ekseni isimlendirme
 plt.xlabel("x isim")
# y ekseni isimlendirme
 plt.ylabel("y isim")
 plt.grid()


######################
# Subplot / birden fazla görsel
######################

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1) # Bir satırlık 3 sütunluk grafik oluştur. İlk grafiği veriyorum.
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([ 50,  60,  70,  80,  90, 100, 110, 120, 130, 140])
y = np.array([80, 82, 84, 86, 88, 90, 92, 94, 96, 98])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([20, 24, 28, 32, 36, 40, 44, 48, 52, 56])
y = np.array([300, 305, 310, 315, 320, 325, 330, 335, 340, 345])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)


######################################################
# SEABORN
######################################################

# High Level

######################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()

# seaborn
sns.countplot(x=df["sex"], data=df)

# matplotlib
df["sex"].value_counts().plot(kind="bar")

######################################
# Sayısal Değişken Görselleştirme
######################################

sns.boxplot(x=df["total_bill"])

df["total_bill"].hist() #pandas



notlar = [68, 74, 82, 90, 78, 85, 92, 88, 76, 61, 79, 73, 89, 81, 72, 95, 70, 83, 77, 75]

plt.hist(notlar, bins=10, edgecolor='r', alpha=0.7)
plt.xlabel('Notlar')
plt.ylabel('Frekans')
plt.title('Sınav Notları Dağılımı')
plt.show()