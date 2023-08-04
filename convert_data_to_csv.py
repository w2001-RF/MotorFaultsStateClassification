import csv
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Liste des noms de fichiers d'images BMP
folder_names = ["A&B50", "A&C&B10", "A&C&B30",
                "A&C10", "A&C30", "A10",
                "A30", "A50",
                "Fan", "Noload", "Rotor-0"]

# Chemin du dossier contenant les images BMP
base_path = "C:\\Users\\DELL\\Desktop\\OCP_stage\\Projet\\IR_Dataset"

# Fonction pour convertir une image BMP en chaîne de pixels
def convert_bmp_to_pixels(file_name):
    try:
        # Ouverture de l'image
        image = Image.open(file_name)
        pixels = np.array(image)
        height, width, channels = pixels.shape

        # Remodelage du tableau de pixels en un tableau 1D
        pixels = pixels.reshape(height * width, channels)

        # Conversion des valeurs des pixels en chaînes de caractères
        pixels = pixels.astype(str)

        # Jointure des valeurs des pixels avec des espaces
        pixel_values = ' '.join([' '.join(pixel) for pixel in pixels])

        return pixel_values
    except Exception as e:
        # Gestion des erreurs lors de la conversion de l'image
        print(f"Erreur lors de la conversion de l'image : {file_name}")
        print(f"Erreur : {str(e)}")
        return None

def get_files_in_directory(directory_path):
    file_list = []
    for root, directories, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    return file_list

# Création du fichier CSV
with open('data.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['short_circuit_faults', 'Pixels'])

    # Utiliser tqdm pour afficher une barre de progression
    for folder_name in tqdm(folder_names, desc='Processing folders'):
        folder_path = os.path.join(base_path, folder_name)
        for file_path in tqdm(get_files_in_directory(folder_path), desc='Processing files'):
            # Conversion de l'image BMP en chaîne de pixels
            pixels = convert_bmp_to_pixels(file_path)

            # Écriture des valeurs dans une nouvelle ligne du fichier CSV
            writer.writerow([folder_name, pixels])

print("Le fichier CSV a été créé avec succès.")


