import json
import os

# Visų galimų požymių sąrašas (informacijai)
# RR_l_0;RR_l_0/RR_l_1;RR_l_1;RR_l_1/RR_l_2;RR_l_2;RR_l_2/RR_l_3;RR_l_3;RR_l_3/RR_l_4
# RR_r_0;RR_r_0/RR_r_1;RR_r_1;RR_r_1/RR_r_2;RR_r_2;RR_r_2/RR_r_3;RR_r_3;RR_r_3/RR_r_4
# seq_size;signal_mean;signal_std;wl_side;wr_side
# P_val;Q_val;R_val;S_val;T_val
# P_pos;Q_pos;R_pos;S_pos;T_pos

eksperimentai = {
    # 1. Mūsų visada naudoti požymiai.
    "1_musu_pozymiai": [
        "Q_val", "R_val", "S_val", "RR_l_0/RR_l_1", "signal_std", "seq_size"
    ],

    # 2. TIK AMPLITUDĖS (Morfologija)
    # Hipotezė: Aritmija keičia EKG bangos formą (aukščius/gylius), nepriklausomai nuo laiko.
    "2_morfologinis_amplitudes": [
        "P_val", "Q_val", "R_val", "S_val", "T_val"
    ],

    # 3. TIK RITMAS (Laikiniai - RR intervalai)
    # Hipotezė: Aritmija yra ritmo sutrikimas. Svarbu tik laikas tarp dūžių.
    "3_laikinis_ritmas_raw": [
        "RR_l_0", "RR_r_0", "RR_l_1", "RR_r_1", "RR_l_2", "RR_r_2"
    ],

    # 4. RITMO SANTYKIAI (Dinamika)
    # Hipotezė: Svarbu ne absoliutus laikas, o kaip intervalas keičiasi lyginant su praėjusiu (pvz., kompensacinė pauzė).
    "4_laikinis_ritmas_santykiai": [
        "RR_l_0/RR_l_1", "RR_l_1/RR_l_2",
        "RR_r_0/RR_r_1", "RR_r_1/RR_r_2"
    ],

    # 5. TIK STATISTINIAI (Globalūs signalo parametrai)
    # Hipotezė: Bendras signalo "energingumas" (std) ir ilgis gali atskirti klases be detalių bangų.
    "5_statistinis_bendras": [
        "signal_mean", "signal_std", "seq_size", "wl_side", "wr_side"
    ],

    # 6. QRS KOMPLEKSO GEOMETRIJA (Svarbiausia dalis)
    # Hipotezė: QRS kompleksas yra ryškiausia dalis. Svarbu ir aukštis, ir plotis (pozicijos).
    "6_qrs_kompleksas_full": [
        "Q_val", "R_val", "S_val",
        "Q_pos", "R_pos", "S_pos"
    ],

    # 7. HIBRIDINIS (Subjektyviai "geriausi")
    # Hipotezė: Derinys iš artimiausio ritmo (RR_0) ir pagrindinių bangų (R, S) duos geriausią rezultatą.
    "7_hibridinis_optimalus": [
        "RR_l_0", "RR_r_0",       # Artimiausi intervalai
        "R_val", "S_val",         # Pagrindinės amplitudės
        "signal_std",             # Bendras signalo nuokrypis
        "RR_l_0/RR_l_1"           # Ritmo pokytis
    ]
}

failo_pavadinimas = 'pozymiu_rinkiniai.json'

# Įrašome į failą
with open(failo_pavadinimas, 'w', encoding='utf-8') as f:
    json.dump(eksperimentai, f, indent=4)

print(f"✓ Failas '{failo_pavadinimas}' sėkmingai sukurtas.")
print(f"Sukurti {len(eksperimentai)} skirtingi rinkiniai.")