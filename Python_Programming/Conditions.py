##############################################################
# KOŞULLAR
##############################################################


# True - False
1 == 1
1 == 2


# if
if 1 == 1:
    print("something")


number = 10

if number == 10:
    print("number is 10")


def num_check(number):
    if number == 10:
        print("number is 10")

num_check(10)


# else

def num_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

num_check(12)


# elif

def num_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

num_check(11)



##############################################################
#  DÖNGÜLER
##############################################################

# for loop

students = ["Sevinç", "Semih", "Ali", "Ayşe"]

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(int(salary*20 / 100 + salary))

for salary in salaries:
    print(int(salary*50 / 100 + salary))

def new_salary(salary, rate):
    return int(salary*rate / 100 + salary)

new_salary(1500,20)

for salary in salaries:
    print(new_salary(salary, 10))


for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


# Uygulama - Mülakat Sorusu

# before: "hi my name is john and i am learning python"
# çift indekstekileri küçüklr tekleri büyült

range(len("miuul"))

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is john and i am learning python")



#############################
# break & contiune & while
#############################

salaries = [1000, 2000, 3000, 4000, 5000]

#break
for salary in salaries:
    if salary == 3000:
        break
    print(salary)

#continue
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

#while

number = 1
while number < 5:
    print(number)
    number += 1


#############################
# Enumerate: Otomatik Counter/Indexer ile for loop
#############################

students = ["Sevinç", "Semih", "Ali", "Ayşe"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)



# Uygulama - Mülakat Sorusu

students = ["Sevinç", "Semih", "Ali", "Ayşe"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return(groups)

divide_students(students)


# Alternating fonksiyonunun enumarete ile yazılması

def alternating_with_enumerate(string):
    new_string = ""
    for index, string in enumerate(string):
        if index % 2 == 0:
            new_string += string.upper()
        else:
            new_string += string.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")


#############################
# Zip
#############################

students = ["Sevinç", "Semih", "Ali", "Ayşe"]

departments = ["IT", "Quality", "HR", "Planing"]

ages = [24, 28, 22, 30]

list(zip(students, departments, ages))



#############################
# lambda, map, filter, reduce
#############################

# lambda
def summer(a, b):
    return a + b

summer(2, 6) * 9

new_sum = lambda a, b: a + b

new_sum(4, 5)

# map

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salar(x):
    return x * 20 /100 + x

for salary in salaries:
    print(new_salar(salary))

list(map(new_salar, salaries))

list(map(lambda x: x* 20 / 100 + x, salaries))

# filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# reduce

from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)
