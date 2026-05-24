# Metodologjia

## Përzgjedhja e domain-it

Domain-i i përzgjedhur për këtë projekt është analiza e imazheve mjekësore, konkretisht klasifikimi i retinopatisë diabetike nga imazhet fundus të retinës. Retinopatia diabetike është një komplikacion serioz i diabetit dhe mund të shkaktojë dëmtim të shikimit nëse nuk identifikohet me kohë. Për këtë arsye, analiza automatike e imazheve fundus me metoda të machine learning dhe deep learning është një drejtim i rëndësishëm kërkimor.

Në këtë projekt fokusi nuk është ndërtimi i një sistemi klinik final, por ndërtimi i një pipeline-i eksperimental, të riprodhueshëm dhe të shpjegueshëm, i cili mund të përdoret për analizë akademike në një kurs të metodologjisë së kërkimit.

## Përzgjedhja e dataset-it

Për eksperimentet u përdor dataset-i **APTOS 2019 Blindness Detection**, i publikuar në Kaggle. Ky dataset përmban imazhe fundus të retinës dhe etiketa diagnostike për pesë nivele të retinopatisë diabetike. Dataset-i është i përshtatshëm për këtë projekt sepse është publik, i strukturuar qartë dhe përdoret shpesh në eksperimente të klasifikimit të retinopatisë diabetike.

Në projekt përdoret vetëm `train.csv` dhe folderi `train_images/`, sepse imazhet publike të testimit në Kaggle nuk kanë etiketa të disponueshme. Prandaj, të dhënat e trajnimit ndahen në mënyrë të kontrolluar në train, validation dhe test.

## Përshkrimi i dataset-it APTOS 2019

Dataset-i përmban 3662 imazhe të etiketuara. Kolona `id_code` identifikon imazhin, ndërsa kolona `diagnosis` përmban etiketën numerike:

```text
0 = No DR
1 = Mild
2 = Moderate
3 = Severe
4 = Proliferative DR
```

Këto etiketa kanë renditje natyrore, sepse paraqesin nivele në rritje të ashpërsisë së sëmundjes. Kjo e bën problemin jo vetëm një klasifikim shumeklasësh, por edhe një problem grading me karakter ordinal.

## Analiza fillestare e të dhënave

Analiza fillestare u krye për të verifikuar strukturën e dataset-it, ekzistencën e imazheve dhe shpërndarjen e klasave. Skripti `01_check_dataset.py` lexon `train.csv`, raporton formatin e dataset-it, kontrollon nëse çdo imazh ekziston në `train_images/` dhe ruan një përmbledhje në `outputs/metrics/dataset_check_summary.csv`.

Skripti `02_eda.py` krijon grafikë për shpërndarjen absolute dhe relative të klasave, ruan shembuj imazhesh sipas klasës dhe llogarit statistika bazike të dimensioneve për një subset të menaxhueshëm imazhesh. Kjo qasje është e përshtatshme për mjedise me burime kompjuterike të kufizuara, sepse nuk kërkon analizimin e të gjitha imazheve në memorie.

## Problemi i class imbalance

Dataset-i është i pabalancuar. Klasa `No DR` përbën rreth gjysmën e të dhënave, ndërsa `Severe` përbën vetëm rreth 5.27%. Ky pabalancim është problematik sepse një model mund të mësojë të favorizojë klasën dominante dhe të japë performancë të dobët në klasat e rralla.

Në këtë projekt class imbalance trajtohet në disa nivele:

- përdoret stratified splitting për të ruajtur shpërndarjen e klasave në train, validation dhe test;
- baseline CNN trajnohet me class weights;
- EfficientNetB0 i përmirësuar përdor balanced sampling;
- përdoret focal loss për të rritur ndikimin e shembujve më të vështirë;
- metrikat macro raportohen krahas metrikave weighted, sepse macro metrics u japin peshë të barabartë të gjitha klasave.

## Preprocessing

Përveç preprocessing bazë, projekti përdor një hap të dedikuar për imazhe fundus. Skripti `08_preprocess_fundus_images.py` krijon imazhe të përpunuara në `data/processed/fundus_224/`.

Ky preprocessing përfshin:

- konvertim në RGB;
- heqje opsionale të kufijve të zinj;
- CLAHE për përmirësim të kontrastit lokal;
- resize në 224x224;
- ruajtje të imazheve të përpunuara në format PNG.

Ky hap është i dobishëm sepse imazhet fundus shpesh kanë sfond të zi, ndriçim jo të njëtrajtshëm dhe kontrast të ndryshëm. Standardizimi i tyre e bën trajnimin më stabil dhe më të përshtatshëm për një model të lehtë si EfficientNetB0.

## Ndarja train/validation/test

Të dhënat ndahen në raport 70/15/15. Ndarja realizohet me `random_state=42`, në mënyrë që rezultatet të jenë të riprodhueshme. Përdoret stratified split sipas kolonës `diagnosis`, me qëllim që çdo pjesë të ruajë sa më shumë të jetë e mundur shpërndarjen e klasave.

Ndarja finale është:

```text
Train: 2563 imazhe
Validation: 549 imazhe
Test: 550 imazhe
```

Test split përdoret vetëm për vlerësimin final, ndërsa validation split përdoret gjatë trajnimit për EarlyStopping dhe ModelCheckpoint.

## Modeli baseline CNN

Modeli baseline CNN është ndërtuar si një arkitekturë e thjeshtë konvolucionale. Ai përmban blloqe `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `GlobalAveragePooling2D`, `Dropout` dhe një dalje `softmax` me pesë klasa.

Qëllimi i këtij modeli nuk është të arrijë performancën më të lartë, por të shërbejë si pikë krahasimi. Nëse një model i thjeshtë trajnohet nga zero dhe performon dobët në klasat minoritare, kjo e justifikon përdorimin e transfer learning dhe teknikave më të avancuara.

Baseline CNN u trajnua me:

- class weights;
- Adam optimizer;
- sparse categorical crossentropy;
- EarlyStopping;
- ModelCheckpoint;
- ReduceLROnPlateau;
- 10 epochs maksimum.

## Modeli EfficientNetB0 me transfer learning

Modeli kryesor përdor EfficientNetB0 me pesha ImageNet. Fillimisht backbone mbahet i ngrirë dhe trajnohet vetëm classification head. Më pas aplikohet fine-tuning i lehtë në shtresat e fundit të EfficientNetB0.

Classification head përbëhet nga:

- `GlobalAveragePooling2D`;
- `Dropout`;
- `Dense` me aktivim `softmax` për pesë klasa.

Versioni i përmirësuar i EfficientNetB0 përdor:

- preprocessing fundus me crop dhe CLAHE;
- balanced sampling për të rritur praninë e klasave minoritare në batch-e;
- focal loss për t'i dhënë më shumë peshë shembujve të vështirë;
- fine-tuning të 20 layer-ave të fundit;
- EarlyStopping dhe ModelCheckpoint.

Kjo qasje mbetet e lehtë krahasuar me modele shumë të mëdha si Vision Transformers, por jep performancë shumë më të mirë se CNN baseline.

## Formulimi ordinal

Për shkak se etiketat 0-4 përfaqësojnë shkallë të rritjes së ashpërsisë, u testua edhe një formulim ordinal. Në vend që modeli të parashikojë pesë klasa të pavarura, ai parashikon katër pragje:

```text
diagnosis > 0
diagnosis > 1
diagnosis > 2
diagnosis > 3
```

Klasa finale merret duke numëruar sa pragje janë parashikuar si të vërteta. Kjo qasje respekton natyrën e renditur të problemit. Në rezultatet finale, modeli ordinal arriti QWK të mirë, por nuk e tejkaloi EfficientNetB0 enhanced në macro F1.

## Cross-validation

Për të adresuar problemin e raportimit të performancës në vetëm një split, projekti përfshin edhe cross-validation. Për shkak se full deep learning cross-validation do të kishte kosto të lartë kompjuterike, u përdor një qasje e përshtatur për burime të kufizuara:

1. EfficientNetB0 i ngrirë përdoret për të nxjerrë feature nga të gjitha imazhet.
2. Feature matrix ruhet në `outputs/metrics/`.
3. Përdoret `StratifiedKFold` me 3 folds.
4. Në çdo fold trajnohet një `LogisticRegression` me `class_weight="balanced"`.

Kjo qasje nuk e zëvendëson plotësisht k-fold deep learning, por jep një matje praktike të stabilitetit të përfaqësimeve vizuale të EfficientNetB0.

## Vlerësimi i performancës

Modelet vlerësohen në test split me metrikat:

- accuracy;
- precision macro dhe weighted;
- recall macro dhe weighted;
- F1 macro dhe weighted;
- classification report;
- confusion matrix;
- normalized confusion matrix;
- Quadratic Weighted Kappa;
- ROC-AUC one-vs-rest kur është e mundur.

Macro metrics janë thelbësore sepse dataset-i është i pabalancuar. QWK është veçanërisht i përshtatshëm sepse penalizon më shumë gabimet e largëta në shkallën ordinal. Për shembull, gabimi `No DR -> Mild` është më pak i rëndë se `No DR -> Proliferative DR`.

## Explainable AI me Grad-CAM

Për interpretueshmëri përdoret Grad-CAM. Kjo metodë krijon heatmap që tregon cilat zona të imazhit kanë ndikuar më shumë në parashikimin e modelit. Në kontekstin e imazheve fundus, Grad-CAM ndihmon për të analizuar nëse modeli fokusohet në zona të retinës që mund të jenë klinikisht relevante.

Skripti `07_gradcam_xai.py` zgjedh raste të sakta dhe të gabuara nga test split, krijon heatmap, i vendos mbi imazhet fundus dhe ruan figurat në:

```text
outputs/gradcam/
outputs/figures/gradcam_examples.png
outputs/metrics/gradcam_selected_cases.csv
```

Grad-CAM nuk provon se modeli është klinikisht i besueshëm, por e bën analizën më transparente dhe më të përshtatshme për diskutim akademik.

## Përshtatshmëria për burime kompjuterike të kufizuara

Projekti është projektuar për mjedis kompjuterik modest. Imazhet nuk ngarkohen të gjitha në memorie, por lexohen batch-by-batch përmes `tf.data`. Madhësia e imazhit është 224x224 dhe batch size mund të konfigurohet. EfficientNetB0 u zgjodh sepse është model transfer learning relativisht i lehtë.

Për të mbajtur projektin praktik:

- preprocessing ruan imazhe 224x224 dhe shmang ripërpunimin e përsëritur;
- batch size 8 përdoret për EfficientNet në konfigurimin eksperimental;
- fine-tuning kufizohet në layer-at e fundit;
- cross-validation bëhet mbi feature të ngrira, jo me trajnime të plota të CNN për çdo fold.

## Research gaps që adresohen nga projekti

Ky projekt adreson disa boshllëqe të rëndësishme të literaturës:

1. Shumë studime raportojnë performancë në një dataset të vetëm. Ky projekt shton cross-validation për të matur stabilitetin.
2. Shumë rezultate bazohen vetëm në accuracy. Ky projekt raporton macro F1, QWK, confusion matrix dhe rezultate për klasë.
3. Dataset-et e retinopatisë diabetike janë të pabalancuara. Projekti përdor class weights, balanced sampling dhe focal loss.
4. Klasat e hershme dhe minoritare janë më të vështira. Projekti i analizon veçmas `Mild`, `Severe` dhe `Proliferative DR`.
5. Modelet deep learning shpesh kanë mungesë transparence në procesin e vendimmarrjes. Projekti përdor Grad-CAM për interpretueshmëri vizuale.
6. Grading i retinopatisë është ordinal. Projekti teston edhe një formulim ordinal krahas softmax classification.

Kontributi i projektit nuk është një metodë klinike e re. Kontributi është metodologjik: një pipeline i plotë, i riprodhueshëm dhe i përshtatshëm për raport akademik, që kombinon performancën numerike me interpretueshmëri dhe analizë të pabalancimit të klasave, duke ruajtur kërkesa kompjuterike të arsyeshme.
