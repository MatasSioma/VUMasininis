import pandas as pd
from tabulate import tabulate

df = pd.read_csv('EKG_pupsniu_analize_subalansuota.csv', sep=';')

columns = list(col for col in df.columns if col.lower() != 'label') #Visi stulpeliai be klases ('label')

df_0 = df[df['label'] == 0.0].reset_index(drop=True) #Visi duomenys 0 klasės (normalūs pūpsniai)
df_1 = df[df['label'] == 1.0].reset_index(drop=True) #Visi duomenys 1 klasės (supraventrikulinis prieširdžių virpėjimas)
df_2 = df[df['label'] == 2.0].reset_index(drop=True) #Visi duomenys 2 klasės (ventrikulinis skilvelių virpėjimas)


#Printinam 0 klasės aprašomąją statistiką
print("\n##################### 0 KLASĖ #####################")
for column in columns:
    print("\n################", column.upper(), "################\n")
    print(df_0[column].describe())
    print('Mediana:' ,df_0[column].median())
    print('Moda: ', df_0[column].mode())


#Printinam 1 klasės aprašomąją statistiką
print("\n##################### 1 KLASĖ #####################")
for column in columns:
    print("\n################", column.upper(), "################\n")
    print(df_1[column].describe())
    print('Mediana:' ,df_1[column].median())
    print('Moda: ', df_1[column].mode())



#Printinam 2 klasės aprašomąją statistiką
print("\n##################### 2 KLASĖ #####################")
for column in columns:
    print("\n################", column.upper(), "################\n")
    print(df_2[column].describe())
    print('Mediana:' ,df_2[column].median())
    print('Moda: ', df_2[column].mode())



#Printinam aprašomąją statistiką bendrai
print("\n##################### BENDRAI ######################")
for column in columns:
    print("\n################", column.upper(), "################\n")
    print(df[column].describe())
    print('Mediana:' ,df[column].median())
    print('Moda: ', df[column].mode())