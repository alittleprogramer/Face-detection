# Analiza wieku i płci na podstawie zdjęć

Projekt dotyczy analizy obrazów z wykorzystaniem metod widzenia komputerowego. Celem było automatyczne wykrywanie twarzy oraz klasyfikacja wieku i płci osób znajdujących się na zdjęciach.

W projekcie wykorzystano modele oparte na bibliotece OpenCV oraz sieci neuronowe zapisane w formacie Caffe. Dla każdej wykrytej twarzy określano przedział wiekowy, płeć oraz prawdopodobieństwo predykcji.

Wyniki analizy zapisywane są do plików CSV oraz prezentowane na obrazach wynikowych z naniesionymi etykietami.

Repozytorium zawiera kod źródłowy, przykładowe dane oraz wyniki działania modelu.

## Wykorzystane modele

W projekcie wykorzystano gotowe modele dostępne online, które odpowiadają za detekcję twarzy oraz klasyfikację wieku i płci. Modele te nie były trenowane samodzielnie.

## Jak uruchomić projekt

1. Pobierz modele i umieść je w odpowiednich folderach:

- **face_detector**  
  (deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel)  
  https://github.com/keyurr2/face-detection/tree/master  

- **age_detector**  
  (age_deploy.prototxt, age_net.caffemodel)  
  https://github.com/Isfhan/age-gender-detection  

- **gender_detector**  
  (gender_deploy.prototxt, gender_net.caffemodel)  
  https://github.com/Isfhan/age-gender-detection  

2. Uruchom skrypt:

```bash
python analiza.py


