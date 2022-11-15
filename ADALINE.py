from tkinter import Y
import pandas as pd
import numpy as np
import random 
import math
import time

start = time.time()

def numeroEntradas(df):
    entradas = len(df.axes[0]) #Calcular el numero de Instancias
    return entradas

def numeroSalidas(df):
    salidas = len(df.value_counts()) #Calcular el numero de salidas
    return salidas

def generadorPesos(y: int):
    x = []
    for i in range(y):
        x.append(random.random())
    return x

def media_normalizacion(df): #La normalización se hace restando la media y dividiendo por la desviación estándar para todos los elementos del Dataframe.#
    return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def minmax_normalizacion(df): #El resultado de la normalización resta el valor mínimo de Dataframe y lo divide por la diferencia entre el valor más alto y el más bajo de la columna correspondiente.#
    return (df - df.min()) / ( df.max() - df.min())

def conversionCategorica(df):
    a = np.unique(df) #Con np.unique obtengo los valores unicos que no se repiten
    #Aqui lo que yo tenia pensado era que, por cada valor de la clase comparar, 
    #si el valor en i es igual al valor que tenemos en a en la posicion j, 
    # remplazo lo que tengo en el df en la posicion i por el numero que es j
    #ejemplo: 
    #print(a[0])
    #print((df_Outputs[0]))
    #vemos que es el mismo valor, entonces hacemos la condicion
    #print( (df_Outputs[0]) == (a[0])) 
    for i in (range(len(df))): #A ver aqui tuve que poner range(len()) porque si no, no jala
        for j in range(len(a)):
            if ((df[i]) == (a[j])):
                df= df.replace(df[i],j) #Reemplazo en todo el dataframe el valor viejito por el nuevo, luego lo convierto en funcion u.u, pa que vea mis comentarios todos mensos
    return df

def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return y

def tangencial(x):
    y = (1 - math.exp(-2*x))/(1 + math.exp(-2*x))
    return y

def combinacion_lineal(df_inputs, df_weights):
    comb = []
    for i in range(len(df_inputs)):
        comb.append(df_inputs[i] * df_weights[i])
    res = sum(comb)
    return res

def ajustes_Peso(pesos, cambios):
    for i in range(len(pesos)):
        pesos[i] = pesos[i] + cambios[i]
    return pesos

def cambio(taza_aprendizaje, diferencia, instancia):
    cambios = list()
    for i in range(len(instancia)):
        cambios.append(taza_aprendizaje * diferencia * instancia[i])
    return cambios

def aprendizaje(df_Inputs, weights, x, taza_aprendizaje):
    repetir = True
    while(x<5):
        errorcont = 0
        x = x + 1
        print("ciclo: "+str(x))
        for i in range(len(df_Inputs)):
            #print(i)
            arr = df_Inputs.iloc[i].to_numpy()
            y = combinacion_lineal(arr, weights)
            print("instancia "+str(arr))
            print("comb lineal: "+str(y))
            
            print("Salida obtenida: "+str(y))

            diferencia = df_Outputs[i] - y
            print("Diferencia "+str(diferencia))

            error = pow(df_Outputs[i] - y,2)
            errorcont = error + errorcont
            print("Error: "+str(error))
            #print((errorcont == 0) and (i==max(range(len(df_Inputs)))))
            if error != 0:
                print("AJUSTE DE PESOS EN LA INSTANCIA: "+str(i+1)) 
                cambios = cambio(taza_aprendizaje, diferencia, arr)
                print("CAMBIOS: ")
                print(cambios)
                weights = ajustes_Peso(weights, cambios)
                print(weights)
                break
                
            if ((errorcont == 0) and (i==max(range(len(df_Inputs))))):
                print("PESOS FINALES ENCONTRADOS")
                repetir = False
                break
        print("HERE WE GO AGAIN")
        if(x<5):
            print("HUBO ERROR EN EL CICLO")
            aprendizaje(df_Inputs, weights, x, taza_aprendizaje)
        break
 

caso = 1

if (caso == 1):
    df = pd.read_csv("or.csv")
    #df = df.drop(['Id'], axis=1) #Aqui quitamos el ID porque no nos sirve
    df_Outputs = df.iloc[:, -1] #Separo la clase de los atributos
    df_Inputs = df.drop(df.iloc[:, -1:].columns, axis=1) #Aqui estan los puros atributos
else: 
    if (caso == 2):
        df = pd.read_csv("and.csv")
        #df = df.drop(['Id'], axis=1) #Aqui quitamos el ID porque no nos sirve
        df_Outputs = df.iloc[:, -1] #Separo la clase de los atributos
        df_Inputs = df.drop(df.iloc[:, -1:].columns, axis=1) #Aqui estan los puros atributos
    else:
        if (caso == 3):
            df = pd.read_csv("libro_ejercicio1.csv")
            #df = df.drop(['Id'], axis=1) #Aqui quitamos el ID porque no nos sirve
            df_Outputs = df.iloc[:, -1] #Separo la clase de los atributos
            df_Inputs = df.drop(df.iloc[:, -1:].columns, axis=1) #Aqui estan los puros atributos

#Inserto el BIAS
df_Inputs.insert(0, "BIAS", 1, allow_duplicates=False)

N_Input = numeroEntradas(df_Inputs)
N_Output = numeroSalidas(df_Outputs)
weights = generadorPesos(len(df.axes[1]))
weights_dummy = [0.35, 0.68, 0.05]

print("Pesos Iniciales")
print(weights_dummy)

#print(df_Inputs)
#print(df_Outputs)

                          
aprendizaje(df_Inputs, weights_dummy, 0, 0.1)
print("Pesos Finales")
print(weights_dummy)

print("Tiempo de ejecucion:")
print("--- %s seconds ---" % (time.time() - start))
