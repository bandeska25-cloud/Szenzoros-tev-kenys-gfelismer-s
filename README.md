# Szenzoros tevékenységfelismerés mélytanulással (UCI HAR)

**Készítette:** Molnár Ádám  
**Tárgy:** Mély tanulás – Féléves projektfeladat  

## 1. Projekt célja
Emberi napi mozgásformák osztályozása okostelefon inerciális szenzorainak (3 tengelyes gyorsulásmérő és 3 tengelyes giroszkóp) 50 Hz-en mintavételezett, 2,56 másodperces (128 időlépéses) idősoros adatai alapján.

**Felismert osztályok (6 kategória):**
- Dinamikus tevékenységek: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`
- Statikus testhelyzetek: `SITTING`, `STANDING`, `LAYING`

## 2. Adathalmaz és előfeldolgozás
- **Forrás:** UCI Human Activity Recognition (HAR) Dataset (30 alany, 9 csatornás nyers szenzorjelek).
- **Adatszivárgás-mentes felosztás:** Személyek (Subject ID) alapján szétválasztott halmazok (Train: 17 alany, Validation: 4 alany, Test: 9 alany).
- **Saját jellemzőkinyerés (47 feature):** Tengelyenkénti statisztikák, jelenergia, Signal Magnitude Area (SMA), eredő vektor-magnitúdó, tengelyközi korrelációk, valamint FFT-alapú domináns frekvencia és spektrális entrópia.

## 3. Kipróbált modellek és eredmények

| Modell | Bemenet | Paraméterszám | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline MLP** | 47 saját jellemző | 1 734 | 89.89% | 0.8965 | 86.53% | 0.8613 |
| **Deep MLP** | 47 saját jellemző | 15 558 | 89.51% | 0.8915 | – | – |
| **1D-CNN** | Nyers idősor (128×9) | 37 062 | 95.13% | 0.9497 | – | – |
| **CNN-BiLSTM (Hangolt)** | Nyers idősor (128×9) | 113 862 | 95.13% | 0.9501 | **91.38%** | **0.9147** |

## 4. Repozitórium felépítése
- `src/preprocessing.py` – Adatletöltés, alany-alapú felosztás, 47 saját jellemző kinyerése és skálázás.
- `src/train_models.py` – Baseline MLP és a hangolt 1D-CNN + BiLSTM modell tanítása és mentése.
- `src/evaluate.py` – A mentett modell kiértékelése a teszthalmazon.
- `figures/` – A projekt során készült elemzési ábrák és tanulási görbék.
