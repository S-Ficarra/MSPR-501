"""
Module get_brut_data

Ce module télécharge des fichiers zip contenant des données depuis des URL, extrait les fichiers CSV
et les enregistre dans un répertoire local pour un traitement ultérieur.

Fonctions principales :
- get_brut_data : Télécharge et extrait les données.

Exemple d'utilisation :
    >>> from get_brut_data import get_brut_data
    >>> get_brut_data()
"""

import io
import os
import zipfile
import requests # type: ignore

from spark.spark import spark_session

spark = spark_session()

urls = [
    # "https://www.kaggle.com/api/v1/datasets/download/imdevskp/h1n1-swine-flu-2009-pandemic-dataset",
    # "https://www.kaggle.com/api/v1/datasets/download/imdevskp/corona-virus-report",
    # "https://www.kaggle.com/api/v1/datasets/download/utkarshx27/mpox-monkeypox-data",
    # "https://www.kaggle.com/api/v1/datasets/download/imdevskp/sars-outbreak-2003-complete-dataset",
    # "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-daily-data.csv"
]

def get_brut_data():
    """
    Télécharge des fichiers zip depuis une liste d'URL, extrait les fichiers CSV 
    et les enregistre dans un dossier local.

    Étapes principales :
    - Crée un dossier local pour stocker les fichiers s'il n'existe pas.
    - Télécharge les fichiers zip depuis les URL spécifiées.
    - Extrait les fichiers CSV contenus dans chaque archive zip.
    - Stocke les fichiers CSV extraits dans des sous-dossiers spécifiques.

    Exceptions :
    - Lève une exception si une URL ne peut pas être téléchargée après le timeout défini.
    - Ignore les fichiers qui ne sont pas au format CSV.

    Notes :
    - Utilise un délai d'attente de 10 secondes pour chaque requête HTTP.
    - Affiche des messages d'information et d'erreur pour suivre le déroulement.

    Exemple :
    >>> get_brut_data()
    [INFO] Dossier créé : ./data_files
    [INFO] Archive téléchargée avec succès depuis : <url>
    [INFO] Fichiers contenus dans l'archive :
    [INFO] ['file1.csv', 'file2.csv']
    [INFO] Extraction et lecture du fichier : file1.csv

    Variables globales utilisées :
    - urls : Liste d'URL contenant les archives zip à télécharger.

    Retour :
    - Aucun retour explicite, les fichiers sont stockés localement.
    """
    data_folder = "./data_files"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"[INFO] Dossier créé : {data_folder}")

    for url in urls:
        res = requests.get(url, allow_redirects=True, timeout=10)
        if res.status_code == 200:
            print(f"[INFO] Archive téléchargée avec succès depuis : {url}")

            zip_name = url.rsplit('/', maxsplit=1)[-1]
            zip_folder = os.path.join(data_folder, zip_name)
            if not os.path.exists(zip_folder):
                os.makedirs(zip_folder)
                print(f"[INFO] Dossier créé pour l'archive : {zip_folder}")

            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                print("[INFO] Fichiers contenus dans l'archive :")
                print(f"[INFO] {z.namelist()}")

                for file_name in z.namelist():
                    if file_name.endswith(".csv"):
                        print(f"[INFO] Extraction et lecture du fichier : {file_name}")
                        file_path = os.path.join(zip_folder, file_name)

                        with z.open(file_name) as f:
                            with open(file_path, "wb") as extracted_file:
                                extracted_file.write(f.read())
        else:
            print(f"[ERROR] Échec du téléchargement depuis {url}. Code d'état : {res.status_code}")
