################################################
# PYTHON ALIŞTIRMALAR
################################################


################
# Görev 1
################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4]
type(l)

d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)

t = ("Machine Learning", "Data Science")
type(t)

s = {"Python", "Machine Learning", "Data Science"}
type(s)

################
# Görev 2
################

text = "The goal is to turn data into information, and information into insight."
dir(text)

text = text.upper()
text = text.replace(",", " ")
text = text.replace(".", " ")
text = text.split()

print(text.upper().replace(",", " ").replace(".", " ").upper().split())

################
# Görev 3
################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
dir(lst)

len(lst)
lst[0]
lst[10]
list_data = lst[0:4]
lst.pop(8)
lst.insert(8, "N")
lst.append(12)

################
# Görev 4
################

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

dict.keys()
dict.values()
dict["Ahmet"] = ["Turkey", 24]
dict.update({"Daisy" : ["England", 13]})
dict["Daisy"][1] = 13
dict.pop("Antonio")


################
# Görev 5
################

l = [2, 13, 18, 93, 22]

def func(list):

    even_list = []
    odd_list = []
    for i in range(len(list)):
        if list[i] % 2 == 0:
            even_list.append(list[i])
        else:
            odd_list.append(list[i])
    return even_list, odd_list

even, odd = func(l)


# Çözüm 2

even_list = []
odd_list = []

[even_list.append(i) if i % 2 == 0 else odd_list.append(i) for i in l]


################
# Görev 6
################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

# Çözüm 1 (kendi çözümüm)
muh_fak = ogrenciler[0:3]
tip_fak = ogrenciler[3:]

for i, ogrenci in enumerate(muh_fak, 1):
       print("Mühendislik Fakültesi", i, ". öğrenci:", ogrenci)

for i, ogrenci in enumerate(tip_fak, 1):
       print("Tıp Fakültesi", i, ". öğrenci:", ogrenci)

# Çözüm 2
for i, x in enumerate(ogrenciler):
    if i<3:
        print("Mühendislik Fakültesi", i, ". öğrenci:", x)
    else:
        i -= 2
        print("Tıp Fakültesi", i, ". öğrenci:", x)

################
# Görev 7
################

ders_kodu = ["CMO1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

# Çözüm 1 (kendi çözümüm)

list = list(zip(ders_kodu, kredi, kontenjan))

for i in list:
    print(f"Kredisi {i[1]} olan {i[0]} kodlu dersin kontenjanı {i[2]} kişidir.")

# Çözüm 2

for ders, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders} kodlu dersin kontenjanı {kontenjan} kişidir.")

################
# Görev 8
################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def func(set1, set2):
    if set1.issuperset(set2):
        return(set1 & set2)
    else:
       return(set2 - set1)

func(kume1, kume2)


################################################
# LİST COMPREHENSİON ALIŞTIRMALAR
################################################


################
# Görev 1
################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.head()


["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

################
# Görev 2
################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

[col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]


################
# Görev 3
################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

ogg_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in ogg_list]
new_df = df[new_cols]
new_df.head()


# HR Sorusu

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)



