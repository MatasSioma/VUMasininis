import json
import os

# ---------- KONSTANTOS ----------
JSON_DIREKTORIJA = "JSON"
JSON_FAILAS = "pozymiu_rinkiniai.json"

# Sukuriame pilną kelią: JSON/pozymiu_rinkiniai.json
FAILO_KELIAS = os.path.join(JSON_DIREKTORIJA, JSON_FAILAS)

# Užtikriname, kad aplankas egzistuoja
os.makedirs(JSON_DIREKTORIJA, exist_ok=True)

eksperimentai = {
    "1_musu_pozymiai": [
        "Q_val", "R_val", "S_val", "RR_l_0/RR_l_1", "signal_std", "seq_size"
    ],

    "2_morfologinis_amplitudes": [
        "P_val", "Q_val", "R_val", "S_val", "T_val"
    ],

    "3_laikinis_ritmas_raw": [
        "RR_l_0", "RR_r_0", "RR_l_1", "RR_r_1", "RR_l_2", "RR_r_2"
    ],

    "4_laikinis_ritmas_santykiai": [
        "RR_l_0/RR_l_1", "RR_l_1/RR_l_2",
        "RR_r_0/RR_r_1", "RR_r_1/RR_r_2"
    ],
    "5_statistinis_bendras": [
        "signal_mean", "signal_std", "seq_size", "wl_side", "wr_side"
    ],

    "6_qrs_kompleksas_full": [
        "Q_val", "R_val", "S_val",
        "Q_pos", "R_pos", "S_pos"
    ],

    "7_hibridinis_optimalus": [
        "RR_l_0", "RR_r_0",
        "R_val", "S_val",
        "signal_std",
        "RR_l_0/RR_l_1"
    ],

    "8_pqrst_kompleksas_full": [
        "P_val", "Q_val", "R_val", "S_val", "T_val",
        "P_pos", "Q_pos", "R_pos", "S_pos", "T_pos"
    ],
}

# Saugome į suformuotą kelią
with open(FAILO_KELIAS, 'w', encoding='utf-8') as f:
    json.dump(eksperimentai, f, indent=4)

print(f"Failas sėkmingai sukurtas: '{FAILO_KELIAS}'")
print(f"Sukurti {len(eksperimentai)} skirtingi rinkiniai.")