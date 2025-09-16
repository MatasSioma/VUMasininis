import pandas as pd

df = pd.read_csv('EKG_pupsniu_analize_subalansuota.csv', sep=';')

columns = list(df.columns) #Visi stulpeliai

for column in ['R_val', 'S_val', 'RR_l_0/RR_l_1', 'signal_mean', 'signal_std', 'seq_size']:
    print("\n################", column.upper(), "################\n")
    print(df[column].describe())
    print('Mediana:' ,df[column].median())
    print('Moda: ', df[column].mode())