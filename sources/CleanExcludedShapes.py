import os
import cv2
import shutil
import re
from skimage.metrics import structural_similarity as ssim

# Dossier des formes à exclure
exclude_dir = r"C:\Users\Sico\Documents\projet\blankFinder\sources\shapes_to_exclude"
similarity_threshold = 0.9  # Seuil de similarité pour supprimer les doublons

import os
import re

# Dossier contenant les images
exclude_dir = r"C:\Users\Sico\Documents\projet\blankFinder\sources\shapes_to_exclude"

# Liste des fichiers dans le dossier
files = os.listdir(exclude_dir)

# Filtrer uniquement les images
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Fonction pour générer un nom valide et standardisé
def sanitize_filename(filename, index):
    valid_name = re.sub(r'[^\w\d_-]', '_', filename)  # Remplace les caractères illisibles par "_"
    return f"image_{index:03d}.jpg"  # Nomme les fichiers en "image_001.jpg", "image_002.jpg", ...


# Renommer les fichiers corrompus
for i, filename in enumerate(image_files, start=1):
    old_path = os.path.join(exclude_dir, filename)
    new_name = sanitize_filename(filename, i)
    new_path = os.path.join(exclude_dir, new_name)

    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("\n✅ Tous les fichiers sont renommés correctement.")

# Charger les images
image_files = [f for f in os.listdir(exclude_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images = {}

for filename in image_files:
    path = os.path.join(exclude_dir, filename)
    img = cv2.imread(path)
    if img is not None:
        images[filename] = img

# Comparer les images et supprimer les doublons
to_remove = set()

for file1 in images:
    img1 = images[file1]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    for file2 in images:
        if file1 == file2 or file2 in to_remove:
            continue  # Ne pas comparer une image avec elle-même

        img2 = images[file2]
        if img1.shape != img2.shape:
            continue  # Ignorer si les tailles sont différentes

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)

        if score >= similarity_threshold:
            print(f"Removing duplicate: {file2} (similar to {file1}, {score:.2%} match)")
            to_remove.add(file2)

# Supprimer les images identifiées comme doublons
for filename in to_remove:
    os.remove(os.path.join(exclude_dir, filename))

print(f"\nCleanup complete. {len(to_remove)} duplicate(s) removed.")
