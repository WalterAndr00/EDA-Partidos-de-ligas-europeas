#%%
#Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import requests
import re
import nltk
import warnings
warnings.filterwarnings('ignore')
from pyod.models.knn import KNN
import time
import seaborn as sns
import math
import pylab 
import scipy.stats as stats
#%%

path="D:\\SoccerGames\\archive\\database.sqlite"
# %%
#Realizar la conexión con SQL
con=sqlite3.connect(path)
cursor=con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

#%%
print(cursor.fetchall())

# %%
#Tabla de partidos
query_match=" SELECT * FROM Match"
data_match=pd.read_sql_query(query_match,con)

#%%
#Tamaño de la tabla de partidos
data_match.shape

#%%
data_match.head()

# %%
#Columnas de la tabla partidos
data_match.columns

# %%
#Tipo de dato de cada variable
data_match.dtypes

# %%
#Primeras columnas a eliminar, ya que son tasas de apuestas
lista1=["B365H","B365D","B365A","BWH","BWD","BWA",
       "IWH","IWD","IWA","LBH","LBD","LBA","PSH",
       "PSD","PSA","WHH","WHD","WHA","SJH","SJD",
       "SJA","VCH","VCD","VCA","GBH","GBD","GBA",
       "BSH","BSD","BSA"]
len(lista1)
#Elimina las columnas con nombres en lista1
def eliminar_columnas(lista):
    for i in lista:
        del(data_match[i])

#%%
eliminar_columnas(lista1)

# %%
#Verificar que se eliminaron las columnas
data_match.shape

# %%
#Que valores compone a la columna shoton
data_match.shoton

# %%
#Elimina las columnas con todos valores vacios
data_match.dropna(axis=1,how='all')

#%%
#Verifico si se eliminaron las columnas
data_match.shape

#%%
#Encuentra los valores nulos
for i in data_match.columns:
    print(i+":",data_match[i].isnull().sum())

#%%
#Elimina los valores nulos
data_match.dropna(how="any",inplace=True)

#%%
#Verificaar que los valores nulos sean eliminados
data_match.isnull().sum()

#%%
data_match.shape
#%%
#%%



#HALLANDO LAS TARJETAS AMARILLAS DE LA COLUMNA CARD

#Primera forma de encotrar las tarjetas.
from nltk.tokenize import word_tokenize


columna_a_tokens=data_match["card"].tolist()
n=len(data_match["card"])
lista_de_listas=[]
for i in range(n):
    palabra=columna_a_tokens[i]
    tokens=word_tokenize(palabra)
    lista_de_listas.append(tokens)

from nltk.probability import FreqDist


K=lista_de_listas[2]
fdist=FreqDist(K)

fdist.most_common(20)

#%%
#Otra forma de encontrar las tarjetas
Prueba=data_match["card"]
type(Prueba)

#%%
Prueba.iloc[2]

#%%
from bs4 import BeautifulSoup

#%%
soup=BeautifulSoup(Prueba.iloc[2],"lxml")
print(soup.prettify())
#%%
#Encuentra las tarjetas amarillas
print(soup.find_all("ycards"))
#Minutos en que cometieron las amarillas
print(soup.find_all("elapsed"))
#Jugadores que recibieron las tarjetas
print(soup.find_all("player1"))
#Equipos al que pertenecen los jugador
print(soup.find_all("team"))

#%%
#Verificar si los anterior es verdad mediante la busqueda del partido en google
#Indica fecha de el partido
print(data_match.iloc[2]["date"])
#Indica el id del equipo que jugo de casa
print(data_match.iloc[2]["home_team_api_id"])
query_team=" SELECT * FROM Team"
data_team=pd.read_sql_query(query_team,con)
data_team.columns
#Indica uno de los equipos donde se recibio la amarilla
print(data_team[data_team["team_api_id"]==8650])
#Los datos son correctos Sounderland Vs Liverpool el dia 2008-08-16


#%%
#Crea un dataframe con 
datos_tarjetas=["ycards"]
tarjetas=pd.DataFrame()
for dato in datos_tarjetas:
    list=[]
    for tag in soup.find_all(dato):
        list.append(tag.get_text())
    tarjetas[dato]=list

#%%
tarjetas["ycards"]=tarjetas["ycards"].astype(int)
#tarjetas["elapsed"]=tarjetas["elapsed"].astype(float)

print(tarjetas["ycards"].sum())
#%%
#Halla las tarjetas amarillas de cada partido
tarjetas_amarillas=[]
for n in range(0,len(Prueba)):
    soup=BeautifulSoup(Prueba.iloc[n],"lxml")
    datos_tarjetas=["ycards"]
    tarjetas=pd.DataFrame()
    for dato in datos_tarjetas:
        list=[]
        for tag in soup.find_all(dato):
            list.append(tag.get_text())
        tarjetas[dato]=list

    tarjetas["ycards"]=tarjetas["ycards"].astype(int)

    tarjetas_amarillas.append(tarjetas["ycards"].sum())

#%%
print(tarjetas_amarillas)

#%%
#Prueba de que que se pueden obtener las tarjetas en menos tiempo
vacia=[]
for i in range(0,data_match.shape[0]):
    soup=BeautifulSoup(Prueba.iloc[i],"lxml")
    vacia.append(len(soup.find_all("elapsed")))

#%%
vacia==tarjetas_amarillas

#%%
#Imprime los indices donde las listas son distintas
ku=0
for i in range(0,len(vacia)):
    if vacia[i]!=tarjetas_amarillas[i]:
        print(i)
        ku=ku+1

#%%
#Número de elementos que son distitos en las listas
ku

#%%
#Verificar algunos partidos donde no coinciden las listas "vacia" y "tajertas_amarillas"
def verif_partidos(x):
    print("Fecha:",data_match.iloc[x]["date"])
    cod=data_match.iloc[x]["home_team_api_id"]
    prename=data_team[data_team["team_api_id"]==cod]
    print("Nombre del equipo:",prename["team_long_name"])
    print("Tarjetas lista 'vacia':",vacia[x])
    print("Tarjetas lista 'tarjetas_amarillas':",tarjetas_amarillas[x])

#%%
verif_partidos(48)
#Efectivamente para dicho partido entre el Tottenham y Blackburn Rovers
# Se cometieron 6 tarjetas amrillas, entre ellas dos a un mismo jugador.

#%%
verif_partidos(164)
#En el partido de Sunderland VS. Stoke City se colocaron 3 amarillas
#y una roja

#%%
verif_partidos(13312)
#En el partido de Fc Sion VS. Fc Zurich se colocaron 3 amarillas
#y una roja

#Conclusion: La lista "vacia" cuenta todas las tarjetas.
#            La lista "tarjetas_amarillas" cuenta solo las tarjetas amarillas.              

#%%
class otras_estadisticas:
    def corners(x):
        corners= data_match["corner"]
        soup3=soup=BeautifulSoup(corners.iloc[x],"lxml")
        return(len(soup3.find_all("elapsed")))
    
    def foul(x):
        fouls= data_match["foulcommit"]
        soup4=BeautifulSoup(fouls.iloc[x],"lxml")
        return(len(soup4.find_all("elapsed")))
    
    def card(x):
        card=data_match["card"]
        soup5= BeautifulSoup(card.iloc[x],"lxml")
        return(len(soup5.find_all("elapsed")))
    
    def cross(x):
        cross=data_match["cross"]
        soup5= BeautifulSoup(cross.iloc[x],"lxml")
        return(len(soup5.find_all("elapsed")))
    
    def shoton(x):
        shoton=data_match["shoton"]
        soup5= BeautifulSoup(shoton.iloc[x],"lxml")
        return(len(soup5.find_all("elapsed")))
    
    def shotoff(x):
        shotoff=data_match["shotoff"]
        soup5= BeautifulSoup(shotoff.iloc[x],"lxml")
        return(len(soup5.find_all("elapsed")))
    
#%%
def imp_todas_estadisticas(x):
    print("Tiros de esquina:",otras_estadisticas.corners(x))
    print("Faltas hechas:",otras_estadisticas.foul(x))
    print("Tarjetas:",otras_estadisticas.card(x))
    print("Pases cruzados:",otras_estadisticas.cross(x))
    print("Tiros a puerta:",otras_estadisticas.shoton(x))
    print("Tiros a fuera:",otras_estadisticas.shotoff(x))
#%%
#Verifico para algún partido dichas estadistica.
imp_todas_estadisticas(48)


#%%
imp_todas_estadisticas(164)

#%%
imp_todas_estadisticas(13312)

#%%
soupp=BeautifulSoup(data_match["possession"].iloc[2],"lxml")
print(soupp.prettify())

#%%
"""Problema: Deben existir valores que no tengan 
las etiquetas que se estan asignando,
probablemente en la pagina de donde se extrajeros 
los datos no estaba dicha información. Así,
procederé a eliminar aquellas inconsistencias"""

#%%
"""Además para el caso de la columna "possession"
 la eliminaré, ya que no me interesa predecir la 
 poseción de cada equipo en los tiempos establecidos"""

#%%
del(data_match["possession"])

#%%
data_match.shape

#%%
listass=[otras_estadisticas.corners,
         otras_estadisticas.foul,
         otras_estadisticas.card,
         otras_estadisticas.cross,
         otras_estadisticas.shoton,
         otras_estadisticas.shotoff]
kkk=["Tiros_esquina","Faltas",
     "Tarjetas","Pases_cruzados",
     "Tiros_a_puerta","Tiros_a_fuera"]

#%%     
def valor(titulo,declase):
    l=[]
    for i in range(0,data_match.shape[0]):
        a=declase(i)
        l.append(a)
    data_match[titulo]=l

#%%
valor("Tiros_esquina",otras_estadisticas.corners)

#%%
valor("Faltas",otras_estadisticas.foul)

#%%
valor("Tarjetas",otras_estadisticas.card)

#%%
valor("Pases_cruzados",otras_estadisticas.cross)

#%%
valor("Tiros_a_puerta",otras_estadisticas.shoton)

#%%
valor("Tiros_a_fuera",otras_estadisticas.shotoff)

#%%

"""for j in range(0,len(kkk)):
    l=[]
    for i in range(0,data_match.shape[0]):
        a=listass[j](i)
        l.append(a)
    data_match[kkk[j]]=l"""
 
#%%
data_match.shape

#%%
for l in kkk:
    serie=data_match[l]
    print(serie[serie==0].count())

#%%
def f_eliminar(x,data):
    filas_a_eliminar=[]
    k=0
    for i in range(0,len(data[x])):
        if data[x].iloc[i]==0:
            filas_a_eliminar.append(i)
    indices=data.index[filas_a_eliminar]
    return(data.drop(indices,axis=0,inplace=True))



#%%
for l in kkk:
    f_eliminar(l,data_match)

#%%
for l in kkk:
    serie=data_match[l]
    print(serie[serie==0].count())

#%%
data_match.shape

#%%
lista2=["cross","goal","shoton","shotoff",
        "foulcommit","card","corner"]

#%%
eliminar_columnas(lista2)

#%%
lista3=data_match.columns[11:55]
#%%
lista3

#%%
eliminar_columnas(lista3)

#%%
data_match.shape

#%%
data_match.columns

#%%
data_match.drop_duplicates()

#%%
data_match.shape

#%%
#LIMPIEZA DE ATRIBUTOS DE LOS EQUIPO
query_team_atr=" SELECT * FROM Team_Attributes "
data_team_atr=pd.read_sql_query(query_team_atr,con)

#%%
data_team_atr.columns
#%%

data_team_atr.isna().sum()

#%%
data_team_atr["buildUpPlayDribbling"].isna().sum()

#%%
len(data_team_atr["buildUpPlayDribbling"])

#%%
del(data_team_atr["buildUpPlayDribbling"])

#%%
data_team_atr.isna().sum()

#%%
data_team_atr.duplicated().sum()

#%%
len(data_team_atr["team_api_id"].unique())

#%%
len(data_team_atr["team_api_id"])

#%%
data_team_atr.iloc[0]

#%%
for i in data_team_atr.columns:
    if "Class" in i:
        del(data_team_atr[i])

#%%
data_team_atr.columns

#%%
data_match["away_team_api_id"].unique()
#%%
data_match[data_match["away_team_api_id"]==10261]

#%%
data_team_atr[data_team_atr["team_api_id"]==10261]

#%%
ids_equi_vis=sorted(data_match["away_team_api_id"].unique())
ids_team=sorted(data_team_atr["team_api_id"].unique())
#%%
print(len(ids_equi_vis))
print(len(ids_team))

#%%
ids_equi_vis
#%%
ids_team
#%%
lista_ids=[]
jj=0
for i in range(0,len(ids_equi_vis)):
    while ids_team[jj]<ids_equi_vis[i]:
        lista_ids.append(ids_team[jj])
        ids_team.pop(jj)
    jj=jj+1
#%%
lista_ids.append(ids_equi_vis[-1])
ids_team.pop(-1)

#%%
len(lista_ids)

#%%
print(len(ids_equi_vis),len(ids_team))

#%%
print(ids_equi_vis==ids_team)

#%%
data_team_atr["team_api_id"]

#%%
data_team_atr.sort_values("team_api_id",inplace=True)

#%%
data_team_atr["team_api_id"]

#%%
for i in range(0,len(lista_ids)):
    data_team_atr.drop(data_team_atr[data_team_atr["team_api_id"]==lista_ids[i]].index,inplace=True)

#%%
len(data_team_atr["team_api_id"].unique())
len(ids_team)

#%%
data_team_atr.sort_values(by=["team_api_id","date"],inplace=True)

#%%
data_team_atr.head()

#%%
data_match.sort_values(by=["home_team_api_id","date"],inplace=True)

#%%
data_match.head()

#%%
lista4=["id","team_fifa_api_id"]
for i in lista4:
    del(data_team_atr[i])

#%%
data_team_atr.rename(columns={"date":"fecha_equipo"},inplace=True)

#%%
data_team_atr.columns

#%%
data_match=data_match.merge(data_team_atr, left_on="home_team_api_id",
                  right_on= "team_api_id")

#%%
data_match.shape

#%%
print(data_match[["fecha_equipo","date"]])
#%%
t=[]
for i in range(0,data_match.shape[0]):
    if int(data_match["fecha_equipo"][i][:4])==2010 :
        if int(data_match["date"][i][:4])>(int(data_match["fecha_equipo"][i][:4])):
            t.append(i)        
    elif int(data_match["fecha_equipo"][i][:4])==2015 :
        if int(data_match["date"][i][:4])<int(data_match["fecha_equipo"][i][:4]):
            t.append(i)  
    elif int(data_match["date"][i][:4])!=int(data_match["fecha_equipo"][i][:4]) :
        t.append(i)

#%%
len(t)
#%%
data_match.drop(t,inplace=True)

#%%
data_match.shape

#%%
data_match.columns

#%%
data_match['defenceTeamWidth']

#%%
data_match.rename(columns={'buildUpPlaySpeed':"Vel.juego_casa",
                           'buildUpPlayPassing':"Contruccion_juego_pases_casa",
                           'chanceCreationPassing': "Oportunidad_pase_casa",
                           'defenceAggression':"Agresividad_def_casa",
                           'defencePressure':"Presión_def_casa",
                           'chanceCreationCrossing':"Oportunidas_cruce_casa",
                           'chanceCreationShooting':"Oportunidas_tiro_casa",
                           'defenceTeamWidth':"Ancho_defensa_casa"},inplace=True)

#%%
del(data_match["team_api_id"])

#%%
del(data_match["fecha_equipo"])

#%%
data_match=data_match.merge(data_team_atr, left_on="away_team_api_id",
                  right_on= "team_api_id")

#%%
data_match.shape

#%%
t2=[]
for i in range(0,data_match.shape[0]):
    if int(data_match["fecha_equipo"][i][:4])==2010 :
        if int(data_match["date"][i][:4])>(int(data_match["fecha_equipo"][i][:4])):
            t2.append(i)        
    elif int(data_match["fecha_equipo"][i][:4])==2015 :
        if int(data_match["date"][i][:4])<int(data_match["fecha_equipo"][i][:4]):
            t2.append(i)  
    elif int(data_match["date"][i][:4])!=int(data_match["fecha_equipo"][i][:4]) :
        t2.append(i)

#%%
data_match.drop(t2,inplace=True)

#%%
data_match.shape

#%%
data_match.rename(columns={'buildUpPlaySpeed':"Vel.juego_visita",
                           'buildUpPlayPassing':"Contruccion_juego_pases_visita",
                           'chanceCreationPassing': "Oportunidad_pase_visita",
                           'defenceAggression':"Agresividad_def_visita",
                           'defencePressure':"Presión_def_visita",
                           'chanceCreationCrossing':"Oportunidas_cruce_visita",
                           'chanceCreationShooting':"Oportunidas_tiro_visita",
                           'defenceTeamWidth':"Ancho_defensa_visita"},inplace=True)
#%%
data_match.rename(columns={"home_team_goal":"Goles_casa",
                           'away_team_goal':"Goles_visita"},inplace=True)

#%%
columnas_new_data=["Goles_casa","Goles_visita",'Tiros_esquina', 
                 'Faltas', 'Tarjetas', 'Pases_cruzados',
                'Tiros_a_puerta', 'Tiros_a_fuera', 'Vel.juego_casa',
       'Contruccion_juego_pases_casa', 'Oportunidad_pase_casa',
       'Oportunidas_cruce_casa', 'Oportunidas_tiro_casa', 'Presión_def_casa',
       'Agresividad_def_casa', 'Ancho_defensa_casa', 'Vel.juego_visita', 'Contruccion_juego_pases_visita',
       'Oportunidad_pase_visita', 'Oportunidas_cruce_visita',
       'Oportunidas_tiro_visita', 'Presión_def_visita',
       'Agresividad_def_visita', 'Ancho_defensa_visita']

#%%
data_new=data_match[columnas_new_data]

#%%
data_new.index=np.array(range(data_new.shape[0]))

#%%
data_new.head(10)

#Analisis exploratorio

#%%
#Valores de los valores representativos
print(data_new.describe())


#%%
columnas=np.array(data_new.columns)
columnas=columnas.reshape(6,4)

#%%
columnas

#%%
#Función para graficar de a 4 diagramas de caja para las variables
def diagramas_caja(x):
    columna_0=columnas[x].reshape(2,2)
    fig1 ,axs1 = plt.subplots(2, 2)
    green_diamond = dict(markerfacecolor='g', marker='D')
    for i in range(0,2):
        for j in range(0,2):
            a=data_new[columna_0[i,j]].index
            axs1[i,j].boxplot(np.array(data_new[columna_0[i,j]]),flierprops=green_diamond)
            axs1[i,j].set_title(columna_0[i,j])

#%%
#Función para graficar de a 4 diagramas de violín de cada variable
def diagramas_violin(x):
    columna_0=columnas[x].reshape(2,2)
    fig1 ,axs1 = plt.subplots(2, 2)
    for i in range(0,2):
        for j in range(0,2):
            a=data_new[columna_0[i,j]].index
            axs1[i,j].violinplot(data_new[columna_0[i,j]].tolist())
            axs1[i,j].set_title(columna_0[i,j])

#%%
diagramas_caja(0)

#%%
diagramas_violin(0)

#%%
diagramas_caja(1)

#%%
diagramas_violin(1)

#%%
diagramas_caja(2)

#%%
diagramas_violin(2)

#%%
diagramas_caja(3)

#%%
diagramas_violin(3)

#%%
diagramas_caja(4)

#%%
diagramas_violin(4)

#%%
diagramas_caja(5)

#%%
diagramas_violin(5)



#%%
"""Se observan valores atípicos 
en la mayoria de variables.
Además, las variables no tienen una 
distribución acordé debido a la represetación
de los diagramas de violín"""
#%%
#Función que elimina los valores atípicos mediante z-score con threshold=3
def eliminar_atipicos(columna):    
    longitud=1
    data_new.index=np.array(range(len(data_new[columna])))
    while longitud>0:
        z = np.abs(stats.zscore(data_new[columna]))
        threshold = 3
        valores_atipicos=np.where(z > 3)[0]
        valores_atipicos=valores_atipicos.tolist()
        data_new.drop(valores_atipicos,inplace=True)
        data_new.index=np.array(range(len(data_new[columna])))
        longitud=len(valores_atipicos)

#%%
#Elimina los valores atípicos de todas las variables
for i in data_new.columns:
    eliminar_atipicos(i)
#%%
diagramas_caja(0)

#%%
#Visualiza el histograma de una de las variables 
sns.distplot(x = data_new["Oportunidas_tiro_casa"], kde = True)
sns.distplot(x = data_new["Oportunidas_tiro_casa"], kde = True)
plt.show

#%%
"""Si se observa el diagrama dicha variable no tendria una
distribución normal"""
#%%
#Realiza una prueba de normalidad mediante el test de shapiro
alpha=0.05
a,b=stats.shapiro(data_new["Oportunidas_tiro_casa"])
print("Statistics",a,"p-value",b)
if alpha>= b:
    print("Se rechaza H0, por tanto no hay normalidad")
else:
    print("No se rechaza H0, existe normalidad")

#%%
#Hace una copia de la variable y le aplica una transformacion log
d=data_new["Oportunidas_tiro_casa"].copy()
d=d.apply(math.log10)

#%%
#Grafica el histograma de la copia de la variable
sns.distplot(x = d, kde = True)
sns.distplot(x = d, kde = True)
plt.show

#%%
"""No tranforma la variabla a una normal, puede ocurrir debido
a la cantidad de elementos cerca al valor 70"""

#%%
def diagramas_histograma(x):
    columna_0=columnas[x].reshape(2,2)
    fig1 ,axs1 = plt.subplots(2, 2)
    for i in range(0,2):
        for j in range(0,2):
            a=data_new[columna_0[i,j]].index
            axs1[i,j].hist(data_new[columna_0[i,j]].tolist(),histtype ='barstacked')
            axs1[i,j].set_title(columna_0[i,j])


#%%
diagramas_histograma(0)

#%%
diagramas_histograma(1)

#%%
diagramas_histograma(2)

#%%
diagramas_histograma(3)

#%%
diagramas_histograma(4)

#%%
diagramas_histograma(5)

#%%
"""La mayoria de las variables tienen un comportamiento 
extraño a diferencia de algunas"""

#%%
#Grafica el digrama de densidad
def diagrama_densidad(i):
    sns.distplot(x = data_new[i], kde = True)
    sns.distplot(x = data_new[i], kde = True)
    plt.show

#%%
"Se observaran los compartamientos de las variables a predecir"
diagrama_densidad("Goles_casa")

#%%
diagrama_densidad("Goles_visita")

#%%
diagrama_densidad("Tiros_esquina")

#%%
diagrama_densidad("Faltas")

#%%
diagrama_densidad("Tarjetas")

#%%
diagrama_densidad("Pases_cruzados")

#%%
diagrama_densidad("Tiros_a_puerta")

#%%
diagrama_densidad("Tiros_a_fuera")

#%%
data_new["Oportunidas_tiro_casa"]
#%%
diagrama_densidad("Oportunidas_tiro_casa")
#%%
alpha=0.05
for i in data_new.columns:
    a,b=stats.shapiro(data_new[i])
    print("Statistics",a,"p-value",b)
    if alpha>= b:
        print(i)
        print("Se rechaza H0, por tanto no hay normalidad")
    else:
        print("No se rechaza H0, existe normalidad")

#%%
#Predicción de Goles de casa
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
#%%
data_atr_pre_gc=["Goles_casa",'Vel.juego_casa',
       'Contruccion_juego_pases_casa', 'Oportunidad_pase_casa',
       'Oportunidas_cruce_casa', 'Oportunidas_tiro_casa', 'Presión_def_casa',
       'Agresividad_def_casa', 'Ancho_defensa_casa', 'Vel.juego_visita', 'Contruccion_juego_pases_visita',
       'Oportunidad_pase_visita', 'Oportunidas_cruce_visita',
       'Oportunidas_tiro_visita', 'Presión_def_visita',
       'Agresividad_def_visita', 'Ancho_defensa_visita']
data_goles_casa=data_new[data_atr_pre_gc]


#%%
for i in data_atr_pre_gc:
    coef, p = kendalltau(data_new["Goles_casa"], data_new[i])
    print("Goles_casa : "+ i + " := ",coef)
#%%
X=data_new[['Vel.juego_casa','Contruccion_juego_pases_casa', 'Oportunidad_pase_casa',
       'Oportunidas_cruce_casa', 'Oportunidas_tiro_casa', 'Presión_def_casa',
       'Agresividad_def_casa', 'Ancho_defensa_casa', 'Vel.juego_visita', 'Contruccion_juego_pases_visita',
       'Oportunidad_pase_visita', 'Oportunidas_cruce_visita','Oportunidas_tiro_visita', 'Presión_def_visita',
       'Agresividad_def_visita', 'Ancho_defensa_visita']]
y=data_new["Goles_casa"]


X_train, X_test, y_train,y_test=train_test_split(X,y,random_state=0)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

n_neighbors=10
knn=KNeighborsClassifier(n_neighbors) 
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
#%%
print(classification_report(y_test, pred))


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

clf = RandomForestClassifier(n_estimators = 100)  
#%%
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test) 

print("Exactitud de Random Forest es ", metrics.accuracy_score(y_test, y_pred)) 
#%%
