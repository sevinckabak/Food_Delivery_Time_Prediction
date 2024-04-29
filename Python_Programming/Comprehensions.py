print("hello")
x = ("semih")
print(type(x))

# List Comprehension
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salaries(x):
    return x * 20 / 100 + x


new_salaries(1000)

null_list = []

for salary in salaries:
    null_list.append(new_salaries(salary))

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salaries(salary))
    else:
        null_list.append(new_salaries(salary * 2))

# 6 satırlık kod dizilimini tek satırda liste halinde çıkarabiliriz!!

[new_salaries(salary * 2) if salary < 3000 else new_salaries(salary) for salary in salaries]

[salary * 2 for salary in salaries]

# if tek başına kullanılacaksa sağda yazılır. Else ile birlikte kullanılacaksa for bloğu sağa geçer.
[salary * 2 for salary in salaries if salary < 3000]

[new_salaries(salary * 2) if salary < 3000 else new_salaries(salary) for salary in salaries]

students = ["John", "Mark", "Venessa", "Maria"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

# Dict Compherension

dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}

dictionary.keys()
dictionary.items()
dictionary.values()

{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}

{k.upper(): v * 2 for (k, v) in dictionary.items()}

# uygulama
# Key'ler orjinal değerler valuelar ise değiştirilmiş değerler

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

# List and Dict Compherension Uygulamaları

# Bir veri setindeki değişken isimlerini değiştirmek

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

# isminde INS olanların başına FLAG olmayanların başına NO_FLAG eklemek istiyoruz.

df.columns = ["FLAG_" + col if "INS" in col else "NON_FLAG_" + col for col in df.columns]

# Amaç: key'i string, value'su bir liste olan sözlük oluşturmak
# Sadece sayısal değişkenler için yapmak istiyoruz
# Output: {'total': ['mean', 'min', 'max', 'var'],
#         'speeding': ['mean', 'min', 'max', 'var'],
#         'alcohol': ['mean', 'min', 'max', 'var'],
#         'not_distracted': ['mean', 'min', 'max', 'var'],
#         'no_previous': ['mean', 'min', 'max', 'var'],
#         'ins_premium': ['mean', 'min', 'max', 'var'],
#         'ins_losses': ['mean', 'min', 'max', 'var']}


import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "var"]

for col in num_cols:
    soz[col] = agg_list

# kısa yol
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

# numpy; sayısal işlemler
# Pandas; numpy üzerine kurulmuş.
# Veri görselleştirme; Matplotlib & Seaborn

# Neden Numpy? - Hız

import numpy as np

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
a * b

# Numpy arrayi oluşturmak

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(5, 2, (3, 4))

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
b = np.random.randint(10, size=(3, 4))
a.ndim
a.shape
a.size
a.dtype
b.shape
b.ndim
b.size
b.dtype

# reshape
np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

a = np.random.randint(10, size=10)
a[0]
a[0:5] #slice
a[0] = 12

b = np.random.randint(10, size=(3,5))
b[0, 0]
b[2, 3] = 12

b[2, 3] = 2.9
# numpy tek bir tip bilgisi tutar. Floatı algılamaz, integer'a çevirir. Bu sebeple hızlıdır.


b[:, 0] #birinci sütun tüm satılar
b[1, :  ] #2. satır tüm sütunlar
b[0:2 , 0:3]

# Fancy Index

v = np.arange(0, 30, 3)
v[1]

catch = [1, 2, 3]

v[catch] # catch'de karşılık gelen değerleri alır.

# Numpy'da koşullu işlemler

v = np.array([1, 2, 3, 4, 5])

#normal döngü

ab = []
for i in v:
    if i < 3:
        ab.append(i)

#numpy ile

v < 3
v[v < 3]
v[v != 3]


# Numpy ile matematiksel işlemler
v = np.array([1, 2, 3, 4, 5])
v / 5
v * 5 / 10
v ** 2
v - 2

np.subtract(v, 2) #çıkarma işlemi
np.add(v, 3) #toplama
np.mean(v) #ortalama
np.sum(v) #tüm elemanların toplamı
np.min(v)
np.max(v)
np.var(v) #varyans

v = np.subtract(v, 2)

# Numpy ile iki bilinmeyenli denklem çözümü
# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = [[5, 1], [1, 3]]
b = [12, 10]

np.linalg.solve(a, b)

sentence = "Merhaba, Dünya!"

for i in range(len(sentence)):
  print(f"{i}. indeksteki karakter: {sentence[i]}")


def best_friend(txt, a, b):
    return txt.count(a+b) == txt.count(a)

best_friend("i found an ounce with my hound", "o", "u")

txt="i found an ounce with my hound"
a="o"
b="u"

list_store= [1,2,3,4]
reduce(lambda a,b: a+b, list_store)
