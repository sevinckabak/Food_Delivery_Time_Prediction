##################################
# Veri Yapılarına Giriş
##################################


##########################################################
# Sayılar: integer
##########################################################

x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

# Sayılar: complex
x = 2j + 1
type(x)

# Sayılar

a = 5
b = 8

a / 2
a ** 2
a / b * 7

# Tipleri değiştirmek

c = 5.85

int(c)
float(a)

int(a * b / c)


# Karakter Dizisi (String)
x = "Hello ai era"
type(x)

print("Sevinç")

# Çok Satırlı Karakter Dizileri

long_str = """ Veri Yapıları: Hızlı Özet,
Sayılar: int, float, complex
String: str,
Boolean (True-False): bool"""

# Karakter dizilerinin elemanlarına erişmek

name = "Sevinç"
name[0]

# Karakter Dizilerinde Slice İşlemi

name[0:2]

long_str[0:10]

# String içerisinde karakter sorgulama

"Veri" in long_str
"veri" in long_str # kçük-büyük harf duyarlılığı var


##########################################################
# String metodları
##########################################################

dir(int) # bu tiple kullanılabilecek metotlar
dir(str)

name = "Sevinç"
type(name)
type(len)

len(name) # boyut bilgisi verir

"miuul".upper()
"SEVİNÇ".lower()

hi = "naber bebek"
hi.replace("b", "s")

hi.split() # öntanımlı değeri boşluk olduğundan boşluklara göre böler

"şıkıdım şıkı ".strip("ş")

"foo".capitalize()

"foo".startswith("f")

##########################################################
# Boolean
##########################################################

True
False

type(False)

5 == 4

##########################################################
# Liste
##########################################################

# Değiştirilebilir
# Sıralıdır. Index işlemi yapılabilir.
# Kapsayıcıdır.


x = ["btx", "th", "mn"]
type(x)

notes = [1, 2, 3, 4]
type(notes)

word = ["a", "b", "c", "d"]

not_num = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_num[1]
not_num[6]
not_num[6][1]
type(not_num[6])
type(not_num[6][1])

notes[0] = 12

not_num[0:4]


# Liste Metotları

dir(notes)

len(notes)
len(not_num)

# append metodu eleman ekler
notes.append(100)

# pop: indexe göre siler
notes.pop(0)

# insert: indexe ekler
notes.insert(2, 52)

##########################################################
# Sözlük
##########################################################

# Değiştirilebilir.
# Sırasızdır (3.7 sonrası sıralı)
# Kapsayıcı

# key - value

x = {"name": "Sevinç", "Age": 24}
type(x)

dic = {"Reg": ["RMSE0", 10],
       "Leb": ["L0BL", 10],
       "Cart": ["CRT", 30]}


"Reg" in dic

dic["Reg"][1]

dic.get("Leb")

# Value Değiştirmek

dic["Reg"] = ["YSA", 20]

dic.keys()
dic.values()

# Tüm çiftlere tuple formatında erişmek

dic.items()

dic.update({"Reg": 11})

dic.update({"RF": 55})

##########################################################
# Demet (Tuple)
##########################################################

# Değiştirilemez
# Sıralıdır
# Kapsayıcıdır

x = ("leblebi", "süt", "şeker")
type(x)

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t = list(t)

t[0] = 99

t = tuple(t)

##########################################################
# Set
##########################################################

# Değiştirilebilir
# Sırasız + Eşsizdir
# Kapsayıcıdır

x = {"düldül", "boncuk", "şimşek"}
type(x)

# difference(): iki kümenin farkı

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1 - set2

set1.difference(set2) # set1'de olup set2'de olmayanlar
set2.difference(set1) # set2'de olup set1'de olmayanlar

# symetric_difference(): İki kümede birbirine göre olmayanlar

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

# intersection(): İki kümenin kesişimi

set1.intersection(set2)
set2.intersection(set1)

set1 & set2

# union(): İki kümenin birleşimi

set1.union(set2)

# isdisjoint(): İki kümenin kesişimi boş mu?

set1.isdisjoint(set2)

# issubset(): Bir küme diğerinin alt kümesi mi?

set1.issubset(set2)

# issuperset(): bir küme diğerini kapsar mı?

set1.issuperset(set2)

