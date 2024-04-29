
fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 256, -8, -4, -2, 5, -9]

# Example for loop solution to add 1 to each number in the list

    # Klasik yöntem

numbers_plus_one = []
for number in numbers:
    numbers_plus_one.append(number + 1)

    # List Comprehensions

numbers_plus_one = [number + 1 for number in numbers]

# Example code that creates a list of all of the list of strings in fruits and uppercases every string
output = []
for fruit in fruits:
    output.append(fruit.upper())

# Exercise 2 - create a variable named capitalized_fruits and use list comprehension syntax to produce output like ['Mango', 'Kiwi', 'Strawberry', etc...]

output_list_comprehension = [fruit.upper() for fruit in fruits]

# Exercise 3 - Use a list comprehension to make a variable named fruits_with_more_than_two_vowels. Hint: You'll need a way to check if something is a vowel.

fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
vowels = "aeiou"

fruits_with_more_than_two_vowels = [fruit for fruit in fruits if sum(letter in vowels for letter in fruit) > 2]

print(fruits_with_more_than_two_vowels)

def count_vowels(str):
    vowels = "aeiou"
    counter = 0
    for letter in str:
        if letter in vowels:
            counter += 1
    return counter


# Exercise 4 - make a variable named fruits_with_only_two_vowels. The result should be ['mango', 'kiwi', 'strawberry']

fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
vowels = "aeiou"

fruits_with_only_two_vowels = [fruit for fruit in fruits if sum(letter in vowels for letter in fruit) == 2]

# Exercise 5 - make a list that contains each fruit with more than 5 characters

fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
vowels = "aeiou"

more_than_5_characters = [fruit for fruit in fruits if len(fruit) > 5]

# Exercise 6 - make a list that contains each fruit with exactly 5 characters

exactly_5_characters = [fruit for fruit in fruits if len(fruit) == 5]

# Exercise 7 - Make a list that contains fruits that have less than 5 characters

less_than_5_characters = [fruit for fruit in fruits if len(fruit) < 5]

# Exercise 8 - Make a list containing the number of characters in each fruit. Output would be [5, 4, 10, etc... ]

fruit_charactes = [len(fruit) for fruit in fruits]

# Exercise 9 - Make a variable named fruits_with_letter_a that contains a list of only the fruits that contain the letter "a"

fruits_with_letter_a = [fruit for fruit in fruits if "a" in fruit]

# Exercise 10 - Make a variable named even_numbers that holds only the even numbers

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 256, -8, -4, -2, 5, -9]

even_number = [number for number in numbers if number % 2 == 0]

# Exercise 11 - Make a variable named odd_numbers that holds only the odd numbers

odd_number = [number for number in numbers if not number % 2 == 0]

# Exercise 12 - Make a variable named positive_numbers that holds only the positive numbers

positive_number = [number for number in numbers if number > 0]

# Exercise 13 - Make a variable named negative_numbers that holds only the negative numbers

negative_number = [number for number in numbers if number < 0]

# Exercise 14 - use a list comprehension w/ a conditional in order to produce a list of numbers with 2 or more numerals

two_or_more_digits = [num for num in numbers if len(str(abs(num))) >= 2]

# Exercise 15 - Make a variable named numbers_squared that contains the numbers list with each element squared. Output is [4, 9, 16, etc...]

number_squared = [number ** 2 for number in numbers]

# Exercise 16 - Make a variable named odd_negative_numbers that contains only the numbers that are both odd and negative.

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 256, -8, -4, -2, 5, -9]

odd_negative_numbers = [number for number in numbers if number < 0 and number % 2 != 0]

# Exercise 17 - Make a variable named numbers_plus_5. In it, return a list containing each number plus five.

numbers_plus_5 = [number + 5 for number in numbers]

# BONUS Make a variable named "primes" that is a list containing the prime numbers in the numbers list. *Hint* you may want to make or find a helper function that determines if a given number is prime or not.

def asal_sayi_mi(x):
        if x <= 1:
            return False
        for i in range(2, int(x ** 0.5) + 1):
            if x % i == 0:
                return False
        return True

primes = [number for number in numbers if asal_sayi_mi(number) == True]

# Exercise
def rps(p1, p2):
    beats = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if beats[p1] == p2:
        return "Player 1 won!"
    if beats[p2] == p1:
        return "Player 2 won!"
    return "Draw!"

rps("rock", "rock")

fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
def remove_every_other(my_list):
    #list = [my_list.pop(index) for index in range(len(my_list)) if not index % 2 == 0]
    return [element for i, element in enumerate(my_list) if i % 2 == 0]
    #print(list)
    pass

def remove(my_list):
    return my_list[::2]

range(len(fruits))

remove_every_other(fruits)
remove(fruits)


# girilen bir sayının çift/tek olmasını bulunuz ve ekrana yazdırınız

def even_odd(x):
    if x % 2 == 0:
        print("Sayı çifttir.")
    else:
        print("Sayı tektir.")

even_odd(12)

even_odd(11)


# girilen bir x sayısının,yine kullanıcıdan istenen bir y sayısına tam bölünüp bölünmediğini kontrol ediniz ve ekranda gösteriniz

def bolme(x, y):
    if x % y == 0:
        print(f"{x} sayısı {y} sayısına tam bölünür.")
    else:
        print(f"{x} sayısı {y} sayısına tam bölünmez.")

bolme(4,2)
bolme(5,2)

# girilen iki sayıdan hangisinin büyük olduğunu bulan uygulama yazınız

def func(x, y):
    if x > y:
        print(f"{x} > {y}")
    else:
        print(f"{y} > {x}")

func(10, 15)

# eğer ortalama(vize*0.40)+(final*0.60)
# 0-30 aralığında FF
# 31-50 DD
# 51-70 cc
# 71-90 BB
# 91-100 AA
# harf notu belirleyen ve ortalama ile birlikte ekranda gösteren uygulamayı yazınız.
# ekrandaki çıktıyı vize notu : {} final notu : {} ortalama : {} harf notu: {} şeklinde gösteriniz.

def vize_hesaplama(vize, final):
    ort = (vize*0.40) + (final*0.60)
    harf_notu = []
    if ort <= 30:
        harf_notu += ["FF"]

    elif ort <= 50 and ort > 30:
         harf_notu += ["DD"]

     elif ort <= 70 and ort > 50:
        harf_notu += ["CC"]

     elif ort <= 90 and ort > 70:
         harf_notu += ["BB"]

     else:
        harf_notu += ["AA"]
    print(f"Vize Notu: {vize} \nFinal Notu: {final} \nOrtama: {ort} \nHarf Notu: {harf_notu}")

vize_hesaplama(56, 85)

# kullanıcıdan kullanıcı ve şifre istedikten sonra ka: admin ve şifre 1234 ise giriş başarılı değilse hangi bilgi hatalıysa onun uyarısını veren bir uygulama yazınız


# kullanıcıdan sipariş etmek istediği kitap sayısı alarak indirim uygulayan ve müşteriye ödemesi gereken tutarı,
# indirim oranını ve indirimsiz fiyatı gösteren uygulama yazınız.Indirim oranları aşağıdadır:
"""
birim fiyatı 10₺
kitap sayısı
20'den az ise %5 indirim
20-50 ise %10
50-100 ise %15
100'den fazla ise %25 indirim
"""



# kullancıdan almak istediği ürünü isteyerek ürünün hangi reyonda olduğunu gösteren bir uygulama yazınız.
"""
domates,biber,patlıcan -> sebze reyonu
parfüm,sampuan,diş macunu -> kozmetik
cep telefonu,bilgisayar,ses sistemleri->teknoloji reyonu
bunlar dışında bir giriş yapılırsa 'ürün bulunmamaktadır'uyarısı verisin

"""

# rasgele 4 haneli bir doğrulama kodu belirleyiniz ve bu değeri gösteriniz.kullanıcıdan bu kodu doğru bir şekilde girmesini isteyiniz doğru giriş yapılama kadar uyarı veriniz


# sayı tahmin uygulaması
# 1-10 arasında rastgele bir sayı üretilir ekranda gösterilmez.kullanıcıdan o sayıyı tahnin etmesi
# istenir.3 kez tahmin etme hakkı olur.hakları bittiğind game over!sayıyı bildiğinde ise
# tebrikler uyarısı verilir.


personel_listesi = [
    {
        "ad": "nur",
        "soyad": "öztürk"
    },
    {
        "ad": "damla",
        "soyad": "kahraman"
    },
    {
        "ad": "mert",
        "soyad": "boylu"
    },
    {
        "ad": "neslihan",
        "soyad": "kaptan yorübulut"
    }
]
# miuul.com->yukarıdaki listede yer alan tüm personellere mail oluşturucu metot yardımıyla mail oluşturunuz ve  listeye ekleyiniz.

# dışarıdan parametre olarak bir dict.listesi olan ve tüm listedeki çalışanlar için mail oluşturup liste olarak dönen metot yazınız


# Examples(Operator, value1, value2) --> output
# ('+', 4, 7) --> 11 ext.

def basic_op1(operator, value1, value2):
    if operator=='+':
        return value1+value2
    if operator=='-':
        return value1-value2
    if operator=='/':
        return value1/value2
    if operator=='*':
        return value1*value2

def basic_op2(operator, value1, value2):
    return eval("{}{}{}".format(value1, operator, value2))

def basic_op3(operator, value1, value2):
    return eval(str(value1) + operator + str(value2))

# eval = expression = "a + b"
# globals_dict = {"a": 10, "b": 20}
# result = eval(expression, globals_dict)  # result will be 30

# Bu örnekte, expression dizesi Python ifadesi a + b'yi temsil eder.
# globals_dict sözlüğü, a ve b değişkenlerine atanan değerleri içerir.
# result değişkeni, expression ifadesi a ve b değişkenlerini kullanarak yürütülerek elde edilen değeri içerir.

basic_op2("*", 8, 9)

# Create a function with two arguments that will return an array of the first n multiples of x.
# count_by(1,10) #should return [1,2,3,4,5,6,7,8,9,10]
# count_by(2,5) #should return [2,4,6,8,10]
def count_by(x, n):
    return list(range(x, (n * x) + 1, x))

count_by(100,5)


# Complete the function which takes two arguments and returns all numbers which are divisible by the given divisor.
# First argument is an array of numbers and the second is the divisor.
# [1, 2, 3, 4, 5, 6], 2 --> [2, 4, 6]

# Çözüm 1
def divisible_by(numbers, divisor):
    divisible = []
    for number in numbers:
        if number % divisor ==0:
            divisible.append(number)
    return divisible

divisible_by([1,2,3,4,5,6], 2)

# Çözüm 2 - list comprehensions
def divisible_by2(numbers, divisor):
    print([number for number in numbers if number % divisor == 0])

divisible_by2([1,2,3,4,5,6], 2)



# Given an array of integers as strings and numbers, return the sum of the array values as if all were numbers.

# Çözüm 1

def sum_mix(arr):
    numb = 0
    for i in arr:
        if type(i) == str:
            numb += int(i)
        else:
            numb += i
    return numb

# Çözüm 2
def sum_mix2(arr):
    return sum(map(int, arr))

sum_mix2([9, 3, '7', '3'])


# Count how often sign changes in array.

def catch_sign_change(lst):
    return sum( (x >= 0) != (y >= 0) for x, y in zip(lst, lst[1:]) )