import os

from PIL import Image
from matplotlib import pyplot as plt


def basic_stat(repertoire_images, sub_path):
    # Répertoire contenant les images
    categorie = []
    valeur = []
    path_in_use = repertoire_images + "/" + sub_path
    categorie.append(sub_path)
    valeur.append(calcul(path_in_use, sub_path))
    graph_barre(categorie, valeur)


def calcul(path_in_use, sub_path):
    total_images = 0
    taille_images = 0
    dimensions_images = {}

    # Parcourir les fichiers dans le répertoire
    for fichier in os.listdir(path_in_use):
        chemin_fichier = os.path.join(path_in_use, fichier)
        # Vérifier si le fichier est une image
        if os.path.isfile(chemin_fichier) and fichier.endswith((".jpeg")):
            try:
                img = Image.open(chemin_fichier)
                total_images += 1
                taille_images += os.path.getsize(chemin_fichier)
                dimensions_images[fichier] = img.size
                img.close()
            except Exception as e:
                print(f"Erreur lors de l'ouverture de {fichier}: {e}")

    # Afficher les statistiques
    print("dossier de :", sub_path)
    print("Nombre total d'images dans le dossier :", total_images)
    return total_images


def affichage_courbe(history):
    # Affichage des courbes de précision et de perte
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Précision sur l\'entraînement')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Précision sur la validation')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perte sur l\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte sur la validation')
    plt.xlabel('Epoch')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()


def graph_barre(categorie, valeur):
    # Créer une liste de couleurs alternant entre bleu et orange
    couleurs = ['blue' if i % 2 == 0 else 'orange' for i in range(len(categorie))]

    # Création du graphique en barres
    plt.bar(categorie, valeur, color=couleurs)

    # Ajout d'un titre et des labels
    plt.title("Nombre d'images par dossier")
    plt.xlabel('Catégories')
    plt.ylabel('Valeurs')

    # Affichage du graphique
    plt.show()