# Përmbledhje e rezultateve

Ky dokument përmbledh rezultatet finale të projektit **Lightweight Deep Learning and Explainable AI for Diabetic Retinopathy Grading Using APTOS 2019 Fundus Images**. Eksperimentet u kryen në konfigurim kompjuterik konservativ, me imazhe 224x224 dhe pipeline `tf.data`, pa ngarkuar të gjitha imazhet në memorie.

Dataset-i u kompletua me të gjitha imazhet e `train_images/`. Më parë mungonin 306 imazhe, por ato u shkarkuan dhe më pas u përdorën në eksperimentet finale.

## Rezultatet e EDA

EDA u krye me dataset-in e plotë APTOS 2019:

```text
python src/01_check_dataset.py
python src/02_eda.py --dimension-sample 500
```

Përmbledhje:

```text
Rreshta në train.csv: 3662
Imazhe të gjetura lokalisht: 3662
Imazhe që mungojnë: 0
Numri i klasave: 5
Raporti majority/minority class: 9.35
```

Output-et kryesore të EDA:

- `outputs/figures/class_distribution.png`
- `outputs/figures/class_distribution_percent.png`
- `outputs/figures/sample_images_by_class.png`
- `outputs/metrics/eda_summary.csv`

EDA tregon qartë se dataset-i është i pabalancuar. Klasa `No DR` dominon, ndërsa `Severe` është klasa më e rrallë.

## Shpërndarja e klasave

Shpërndarja finale e dataset-it:

| Klasa | Emri | Numri | Përqindja |
|---:|---|---:|---:|
| 0 | No DR | 1805 | 49.29% |
| 1 | Mild | 370 | 10.10% |
| 2 | Moderate | 999 | 27.28% |
| 3 | Severe | 193 | 5.27% |
| 4 | Proliferative DR | 295 | 8.06% |

Kjo pabarazi shpjegon pse accuracy nuk mjafton si metrikë kryesore. Në këtë projekt, `macro F1`, `macro recall` dhe `Quadratic Weighted Kappa` janë më të rëndësishme për interpretim akademik.

## Preprocessing

U krijua preprocessing specifik për imazhe fundus:

```text
python src/08_preprocess_fundus_images.py
```

Ky hap:

- largon kufijtë e zinj kur është e mundur;
- aplikon CLAHE për përmirësim të kontrastit lokal;
- konverton imazhet në RGB;
- i standardizon në 224x224;
- i ruan në `data/processed/fundus_224/`.

Rezultati:

```text
Imazhe të përpunuara: 3662
Folder output: data/processed/fundus_224/
```

Ky preprocessing është më i përshtatshëm për fundus images sesa thjesht resize, sepse retina shpesh ka kufij të zinj dhe ndryshime të ndriçimit.

## Ndarja Train/Validation/Test

Ndarja u krijua me stratified split 70/15/15:

```text
python src/03_create_splits.py
```

Rezultati:

```text
Train: 2563 imazhe
Validation: 549 imazhe
Test: 550 imazhe
```

Shpërndarja në test split:

| Klasa | Emri | Numri |
|---:|---|---:|
| 0 | No DR | 271 |
| 1 | Mild | 56 |
| 2 | Moderate | 150 |
| 3 | Severe | 29 |
| 4 | Proliferative DR | 44 |

## Eksperimentet

U testuan katër qasje kryesore:

1. Baseline CNN me class weights.
2. EfficientNetB0 i përmirësuar me preprocessing fundus, balanced sampling, focal loss dhe fine-tuning.
3. EfficientNetB0 me formulim ordinal.
4. Cross-validation të përshtatur për burime kompjuterike të kufizuara, me frozen EfficientNetB0 features dhe Logistic Regression të balancuar.

Komandat kryesore:

```text
python src/04_train_baseline_cnn.py --epochs 10 --batch-size 16

python src/05_train_efficientnet.py --epochs 10 --batch-size 8 --fine-tune --fine-tune-epochs 5 --fine-tune-layers 20 --balanced-sampling --focal-loss --patience 4

python src/09_train_ordinal_efficientnet.py --epochs 10 --batch-size 8 --balanced-sampling --patience 4

python src/10_cross_validation.py --folds 3 --batch-size 8
```

## Rezultatet kryesore

| Modeli | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | QWK | ROC-AUC macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.5418 | 0.3660 | 0.4491 | 0.3497 | 0.5227 | 0.5951 | 0.8407 |
| EfficientNetB0 enhanced | 0.7873 | 0.6346 | 0.6573 | 0.6399 | 0.7908 | 0.8723 | 0.9272 |
| Ordinal EfficientNetB0 | 0.7418 | 0.5889 | 0.5924 | 0.5553 | 0.7394 | 0.8540 | Nuk aplikohet |

Modeli më i mirë është **EfficientNetB0 enhanced**. Ai jep përmirësim të madh ndaj baseline CNN:

```text
Accuracy: 0.5418 -> 0.7873
Macro F1: 0.3497 -> 0.6399
QWK: 0.5951 -> 0.8723
```

Kjo tregon se transfer learning, preprocessing fundus dhe trajtimi më i mirë i class imbalance e rrisin ndjeshëm cilësinë e klasifikimit.

## Rezultatet sipas klasave

### EfficientNetB0 enhanced

| Klasa | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.9778 | 0.9742 | 0.9760 | 271 |
| Mild | 0.4667 | 0.6250 | 0.5344 | 56 |
| Moderate | 0.7519 | 0.6467 | 0.6953 | 150 |
| Severe | 0.4359 | 0.5862 | 0.5000 | 29 |
| Proliferative DR | 0.5405 | 0.4545 | 0.4938 | 44 |

Modeli dallon shumë mirë klasën `No DR`, ndërsa klasat `Mild`, `Severe` dhe `Proliferative DR` mbeten më të vështira. Kjo është e pritshme për shkak të numrit të vogël të shembujve dhe ngjashmërisë vizuale midis stadeve fqinje.

### Baseline CNN

| Klasa | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.9447 | 0.8819 | 0.9122 | 271 |
| Mild | 0.2778 | 0.4464 | 0.3425 | 56 |
| Moderate | 0.2857 | 0.0133 | 0.0255 | 150 |
| Severe | 0.1429 | 0.5172 | 0.2239 | 29 |
| Proliferative DR | 0.1789 | 0.3864 | 0.2446 | 44 |

Baseline CNN është i dobishëm si pikë krahasimi, por performanca e tij në klasat jo-normale është e dobët. Kjo e mbështet arsyetimin pse transfer learning është i nevojshëm.

### Ordinal EfficientNetB0

| Klasa | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No DR | 0.9502 | 0.9852 | 0.9674 | 271 |
| Mild | 0.4000 | 0.5357 | 0.4580 | 56 |
| Moderate | 0.7193 | 0.5467 | 0.6212 | 150 |
| Severe | 0.3125 | 0.6897 | 0.4301 | 29 |
| Proliferative DR | 0.5625 | 0.2045 | 0.3000 | 44 |

Formulimi ordinal arriti QWK të mirë, 0.8540, por ishte më i dobët se EfficientNetB0 enhanced në macro F1. Ai është metodologjikisht interesant sepse respekton renditjen natyrore të klasave 0-4, por në këtë konfigurim nuk u bë modeli më i mirë.

## Analiza e confusion matrix

Confusion matrix për EfficientNetB0 enhanced:

```text
[[264,  6,  1,  0,  0],
 [  4, 35, 17,  0,  0],
 [  2, 28, 97, 14,  9],
 [  0,  0,  4, 17,  8],
 [  0,  6, 10,  8, 20]]
```

Modeli ka performancë shumë të fortë për `No DR`. Gabimet më të rëndësishme ndodhin midis klasave fqinje:

- `Mild` ngatërrohet me `Moderate`;
- `Moderate` ngatërrohet me `Mild`, `Severe` dhe `Proliferative DR`;
- `Proliferative DR` ngatërrohet shpesh me `Moderate` ose `Severe`.

Kjo është tipike për diabetic retinopathy grading, sepse ndryshimi midis stadeve nuk është gjithmonë i prerë vizualisht.

## Cross-validation

Cross-validation u realizua me 3 folds për të matur stabilitetin:

| Fold | Accuracy | Macro F1 | Weighted F1 | QWK |
|---:|---:|---:|---:|---:|
| 1 | 0.7952 | 0.6278 | 0.7936 | 0.8724 |
| 2 | 0.7797 | 0.6139 | 0.7835 | 0.8311 |
| 3 | 0.7918 | 0.6200 | 0.7931 | 0.8560 |

Mesatarja:

```text
Accuracy mean: 0.7889
Accuracy std: 0.0067
Macro F1 mean: 0.6206
Macro F1 std: 0.0057
QWK mean: 0.8532
QWK std: 0.0170
```

Rezultatet janë të qëndrueshme midis folds. Kjo e forcon interpretimin se modeli nuk po jep rezultat të mirë vetëm rastësisht në një split të vetëm.

## Analiza e Grad-CAM

Grad-CAM u gjenerua me:

```text
python src/07_gradcam_xai.py --num-cases 8 --predictions-csv outputs/metrics/predictions_efficientnet_enhanced.csv
```

Output-et:

- `outputs/gradcam/`
- `outputs/figures/gradcam_examples.png`
- `outputs/metrics/gradcam_selected_cases.csv`

U zgjodhën 8 raste:

```text
5 raste të klasifikuara saktë
3 raste të klasifikuara gabim
```

Rastet e sakta përfshijnë shembuj nga të gjitha klasat 0-4. Rastet e gabuara përfshijnë ngatërrime të dobishme për diskutim, si `No DR -> Mild`, `Mild -> No DR` dhe `Moderate -> Severe`.

Grad-CAM nuk e zëvendëson validimin klinik, por ndihmon në diskutimin e interpretueshmërisë: a fokusohet modeli në retinë dhe në zona potencialisht relevante, apo po përdor artefakte?

## Sa të mira janë rezultatet?

Rezultatet janë të përshtatshme për një projekt akademik me konfigurim kompjuterik konservativ. EfficientNetB0 enhanced arrin:

- accuracy të lartë për këtë konfigurim: 78.7%;
- macro F1 dukshëm më të mirë se baseline: 64.0%;
- QWK shumë të mirë: 0.8723;
- ROC-AUC macro të fortë: 0.9272.

Megjithatë, projekti nuk duhet të konsiderohet si sistem klinik i gatshëm për përdorim. Arsyeja kryesore është se performanca në klasat minoritare mbetet e kufizuar. `Severe` ka vetëm 29 raste në test split, ndërsa `Proliferative DR` ka 44, prandaj metrikat për këto klasa janë më të paqëndrueshme.

## Si mund të përmirësohet më tej?

Përmirësime të mundshme:

- përdorimi i validimit në një dataset tjetër, si Messidor ose EyePACS, për të testuar generalizimin;
- rritja e rezolucionit në 299x299 ose 380x380 nëse ka GPU;
- përdorimi i augmentation më të kujdesshëm për fundus images;
- tuning i threshold-eve për formulimin ordinal;
- përdorimi i ensemble midis softmax dhe ordinal model;
- focal loss me alpha të peshuar sipas klasave;
- auditim manual i Grad-CAM nga një person me njohuri klinike;
- testim i preprocessing alternative, për shembull vetëm crop, crop + CLAHE, dhe pa CLAHE;
- përdorimi i k-fold deep learning në GPU, nëse burimet kompjuterike e lejojnë.

## Përfundimi

Modeli më i mirë final është **EfficientNetB0 enhanced**. Ai tejkalon qartë baseline CNN dhe jep rezultat të qëndrueshëm në cross-validation. Projekti shton vlerë metodologjike sepse nuk raporton vetëm një accuracy, por kombinon:

- preprocessing të përshtatur për fundus images;
- trajtim më të mirë të class imbalance;
- krahasim baseline vs transfer learning;
- formulim ordinal;
- cross-validation për stabilitet;
- Grad-CAM për Explainable AI.

Kjo e bën projektin të përshtatshëm për një punim universitar të metodologjisë së kërkimit, edhe pse nuk është një sistem klinik final.
