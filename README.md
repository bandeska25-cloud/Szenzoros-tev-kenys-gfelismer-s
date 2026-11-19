# Szenzoros tevékenységfelismerés mélytanulással (UCI HAR)

**Készítette:** Molnár Ádám  
**Tárgy:** Mély tanulás – Féléves projektfeladat

## 1. Projekt célja

A projekt célja emberi napi mozgásformák osztályozása okostelefon inerciális szenzorainak – háromtengelyes gyorsulásmérő és háromtengelyes giroszkóp – idősoros adatai alapján. A szenzoradatok mintavételezési frekvenciája 50 Hz; egy minta 128 időlépésből, vagyis 2,56 másodpercnyi mérésből áll.

**Felismert osztályok (6 kategória):**

- Dinamikus tevékenységek: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`
- Statikus testhelyzetek: `SITTING`, `STANDING`, `LAYING`

## 2. Adathalmaz és előfeldolgozás

- **Forrás:** UCI Human Activity Recognition (HAR) Dataset, 30 alany és 9 csatornás nyers szenzorjel.
- **Adatszivárgás-mentes felosztás:** az adatok alanyazonosító (Subject ID) szerint különülnek el. A tanítóhalmaz 17, a validációs halmaz 4, a teszthalmaz 9 alanyból áll.
- **Saját jellemzőkinyerés (47 jellemző):** tengelyenkénti statisztikák, jelenergia, Signal Magnitude Area (SMA), eredő vektor magnitúdója, tengelyközi korrelációk, valamint FFT-alapú domináns frekvencia és spektrális entrópia.

Az időbeli leírás a mintavételezési frekvenciára és a mérési ablak hosszára vonatkozik (50 Hz, 128 időlépés = 2,56 másodperc). A projekt nem tartalmaz percre pontos időbélyegeket.

## 3. Kipróbált modellek és eredmények

| Modell | Bemenet | Paraméterszám | Val. accuracy | Val. macro F1 | Teszt accuracy | Teszt macro F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Baseline MLP** | 47 saját jellemző | 1 734 | 89,89% | 0,8965 | 86,53% | 0,8613 |
| **Deep MLP** | 47 saját jellemző | 15 558 | 89,51% | 0,8915 | – | – |
| **1D-CNN** | Nyers idősor (128 × 9) | 37 062 | 95,13% | 0,9497 | – | – |
| **CNN-BiLSTM (hangolt)** | Nyers idősor (128 × 9) | 113 862 | 95,13% | 0,9501 | **91,38%** | **0,9147** |

## 4. Főbb megállapítások

- A nyers idősorokra épülő **1D-CNN + BiLSTM** hibrid architektúra közel **5 százalékponttal** múlta felül a kézzel kinyert jellemzőkre épülő MLP baseline tesztpontosságát.
- A fekvő (`LAYING`) testhelyzetet a modell 100%-os pontossággal ismeri fel.
- A hibák körülbelül 75%-a az ülő (`SITTING`) és álló (`STANDING`) testhelyzetek felcseréléséből adódik; a zsebben tartott telefon hasonló gravitációs orientációja megnehezíti e két osztály megkülönböztetését.

## 5. Projektfájlok

- `har_cnn_bilstm_best.keras` – a megadott projektarchívumból átvett, nyertes CNN-BiLSTM modell.
- `har_raw_scaler.pkl` – a megadott projektarchívumból átvett skálázó.
- `figures/` – a projekthez átadott nyolc ábra; a `06_gyoztes_modell_tanulasi_gorbek.png` fájlnévben a forrásmappa dupla pontját egyetlen pontra javítottuk.
- `HAR_Melytanulas_Projekt.ipynb` – dokumentációs célú, rekonstruált notebook, amely összefoglalja a megadott projektadatokat és megjeleníti az ábrákat. **Nem az eredeti, letöltött Colab tanító notebook:** az eredeti nem volt jelen sem a repóban, sem a megadott forrásmappában/archívumban. A helyettesítő notebook nem tartalmaz előfeldolgozási vagy tanítási kódot, és nem teszi reprodukálhatóvá a kísérleteket.

A `.keras` modell betöltéséhez a mentéssel kompatibilis TensorFlow/Keras környezet szükséges. A skálázó bináris állományának pontos formátuma és betöltési függvénye az eredeti notebook hiányában nem ellenőrizhető; a modell alkalmazásakor ugyanazt az előfeldolgozást és skálázást kell használni, amelyet a tanítás során alkalmaztak. A dokumentációs notebook ábramegjelenítő cellája a repó gyökeréből futtatandó, ahol a `figures/` mappa elérhető.

## 6. Ábrák

1. [EDA – eloszlás és mintajelek](figures/01_eda_eloszlas_es_mintajelek.png)
2. [Saját jellemzők boxplotja](figures/02_sajat_jellemzok_boxplot.png)
3. [Baseline konfúziós mátrix](figures/03_baseline_konfuzios_matrix.png)
4. [Modellek tanulási görbéi](figures/04_modellek_tanulasi_gorbek.png)
5. [Konfidenciaküszöb-eloszlás](figures/05_konfidencia_kuszob_eloszlas.png)
6. [A győztes modell tanulási görbéi](figures/06_gyoztes_modell_tanulasi_gorbek.png)
7. [Végső teszt konfúziós mátrix](figures/07_vegso_konfuzios_matrix_test.png)
8. [Hibaelemzési példák](figures/08_hibaelemzes_peldak.png)
