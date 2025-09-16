import pandas as pd

df = pd.read_csv('EKG_pupsniu_analize.csv', sep=';')

# 1. Žingsnis, atsirinkti duomenys pagal klases 3x500
df_balanced = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=500, random_state=22))

df_balanced = df_balanced.reset_index(drop=True)

# 2. Žingsnis, atsirinkti savo pasirinktus požymius
pozymiai = ['R_val', 'S_val', 'RR_l_0/RR_l_1', 'signal_mean', 'signal_std', 'seq_size', 'label']

df_tik_su_pozymiais = df_balanced[pozymiai]

df_tik_su_pozymiais.to_csv('EKG_pupsniu_analize_subalansuota.csv', index=False, sep=';')