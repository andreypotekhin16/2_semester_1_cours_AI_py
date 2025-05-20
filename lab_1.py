import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

cla_pos_ss1 = 51 # позиция, где кончается первый класс элементов
cla_pos_ss2 = 130 # позиция, где кончается второй класс элементов
cla_pos_ss3 = 178 # позиция, где кончается третий класс элементов

data_name = "wine.data" # название файла с данными
data_type = "f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"  # типы данных и количество элементов в строке файла. 
#                                                                       f8 - числа, U30 - строки

lib = [   #названия колонок, первая - это колонка с классом элементов, по нему они групируются.
    "class",
    "Alcohol",
 	"Malic acid",
 	"Ash",
	"Alcalinity of ash",
 	"Magnesium",
	"Total phenols",
 	"Flavanoids",
 	"Nonflavanoid phenols",
 	"Proanthocyanins",
	"Color intensity",
 	"Hue",
 	"OD280/OD315 of diluted wines",
 	"Proline",
]
#--------------------------------------------внизу уже код--------------------------------------------#

length = len(data_type.split(","))
dt = np.dtype(data_type)
data = np.genfromtxt(data_name, delimiter = ",", dtype = dt )


class DATA:
    def __init__(self):
        self.arr = []

    def input_arr(self, number):
        self.arr = []  # очищаем массив
        for i, dot in enumerate(data):
            self.arr.append(dot[number])
    
#создаём массив объектов класса
names = [DATA() for _ in range(length)]
for i in range(length):
    names[i].input_arr(i)



#вывод
for i in range (1,13):
    
    Ox = names[i].arr
    Oy = names[i+1].arr
    plt.xlabel(lib[i]) 
    plt.ylabel(lib[i+1])
    plt.figure(i)
   
    setosa, = plt.plot(Ox[:cla_pos_ss1], Oy[:cla_pos_ss1], 'ro', label=lib[0] + "1")

    versicolor, = plt.plot(Ox[cla_pos_ss1:cla_pos_ss2], Oy[cla_pos_ss1:cla_pos_ss2], 'g^', label=lib[0] + '2')

    virginica, = plt.plot(Ox [cla_pos_ss2:cla_pos_ss3], Oy[cla_pos_ss2:cla_pos_ss3], 'bs', label=lib[0] + "3")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)

  

plt.show()
