import numpy as np # bibliothèque Python pour le calcul scientifique 
# (travailler avec des tableaux multidimensionnels et effectuer des opérations mathématiques efficaces)
import matplotlib.pyplot as plt # bibliothèque de visualisation de données en Python.
from keras.datasets import mnist # Importation du jeu de données MNIST


class MNISTLoader:
    # methode appellée quand une nouvelle instance de cette classe est crée 
    def __init__(self):
        # Chargement des données MNIST
        # x_train => images des chiffres écrits à la main
        # y_train => étiquettes correspondantes
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
    def preprocess_data(self):
        # Normalisation des données pour les mettre à l'échelle entre 0 et 1
        # images => stockées avec des valeurs de pixels allant de 0 à 255 pour chaque canal de couleurs
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.

        # Redimensionnement des données pour les adapter au modèle (ajout d'une dimension pour le canal de couleur)
        # np.expand_dims() => fonction fournie par la bibliothèque NumPy => ajouter une dimension supplémentaire à un tableau multidimensionnel.
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        
    def show_digit(self, index):
        # Affichage d'un chiffre particulier à l'index spécifié
        digit_image = self.x_train[index].reshape(28, 28)  # Redimensionnement de l'image à sa forme originale
        plt.imshow(digit_image, cmap='gray')  # Utilisation d'une colormap en niveaux de gris
        plt.axis('off')  # Désactivation des axes
        plt.show()
        
    def display_basic_statistics(self):
        # Calculer la distribution des chiffres dans les données d'entraînement
        
        # np.unique => fonction de la bibliothèque NumPy => trouve les éléments uniques dans un tableau.
        # return_counts=True : Lorsque ce paramètre est défini à true, np.unique renvoie deux tableaux :
        # Le premier tableau contient les éléments uniques trouvés dans self.y_train.
        # Le deuxième tableau contient le nombre de fois que chaque élément unique apparaît dans self.y_train.
        unique, counts = np.unique(self.y_train, return_counts=True)
        train_distribution = dict(zip(unique, counts))
        
        # Calculer la distribution des chiffres dans les données de test
        unique, counts = np.unique(self.y_test, return_counts=True)
        test_distribution = dict(zip(unique, counts))
        
        # Afficher les distributions
        # .items() => retourne une vue sur les paires (clé, valeur) du dictionnaire. 
        print("Distribution des chiffres dans les données d'entraînement :")
        for digit, count in train_distribution.items():
            print(f"Chiffre {digit} : {count} occurrences")

        print("\nDistribution des chiffres dans les données de test :")
        for digit, count in test_distribution.items():
            print(f"Chiffre {digit} : {count} occurrences")
            
        
        # Visualiser les distributions avec des graphiques
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(train_distribution.keys(), train_distribution.values(), color='blue')
        axes[0].set_title('Distribution des chiffres dans les données d\'entraînement')
        axes[0].set_xlabel('Chiffres')
        axes[0].set_ylabel('Occurrences')

        axes[1].bar(test_distribution.keys(), test_distribution.values(), color='green')
        axes[1].set_title('Distribution des chiffres dans les données de test')
        axes[1].set_xlabel('Chiffres')
        axes[1].set_ylabel('Occurrences')

        plt.show()
    
    def display_image(self, index, dataset='train'):
        """
        Affiche une image spécifique d'un ensemble de données choisi.

        Parameters:
        - index (int): l'index de l'image à afficher.
        - dataset (str): 'train' pour l'ensemble d'entraînement, 'test' pour l'ensemble de test.
        """
        if dataset == 'train':
            image = self.x_train[index]
            label = self.y_train[index]
        elif dataset == 'test':
            image = self.x_test[index]
            label = self.y_test[index]
        else:
            raise ValueError("Dataset must be 'train' or 'test'")

        # Si l'image a une dimension de canal, la réduire pour l'affichage
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()
        
    def display_mean_digits(self):
        # Calculer la moyenne de chaque chiffre dans les données d'entraînement
        mean_digits = []
        for digit in range(10):
            # Sélectionner toutes les images correspondant à ce chiffre
            digit_images = self.x_train[self.y_train == digit]
            # Calculer la moyenne de ces images
            mean_image = np.mean(digit_images, axis=0) # calculer la moyenne des valeurs de pixels pour chaque pixel de toutes les images du chiffre.
            #  nombre d'images de 5, 28, 28). 
            # L'axe 0 de digit_images parcourt les différentes images de 5.
            # L'axe 1 représente la hauteur des images (28 pixels de haut).
            # L'axe 2 représente la largeur des images (28 pixels de large).
            mean_digits.append(mean_image) #  ajoute l'image moyenne calculée précédemment à cette liste.

        # Afficher les moyennes des chiffres
        fig, axes = plt.subplots(2, 5, figsize=(12, 6)) # crée une nouvelle figure Matplotlib avec une grille de 2 lignes et 5 colonnes, pour un total de 10 sous-graphiques.
        # figsize => la largeur et la hauteur en pouces de la zone dans laquelle les sous-graphiques seront affichés.
        for i in range(10):
            ax = axes[i // 5, i % 5]
            ax.imshow(mean_digits[i], cmap='gray')
            ax.set_title(f"Chiffre {i}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        
    def reshape_images(self):
        # Remodeler les images pour les adapter à la taille [n, 784]
        self.x_train = self.x_train.reshape(-1, 28*28)
        self.x_test = self.x_test.reshape(-1, 28*28)
        
        print(f" Train: {self.x_train.shape }   Test: {self.x_test.shape }")


        
# Création d'une instance de la classe MNISTLoader
mnist_loader = MNISTLoader()

# # Prétraitement des données
# mnist_loader.preprocess_data()

# # Affichage d'un exemple de chiffre
# index_to_show = 5 # Indice de l'image à afficher
# mnist_loader.show_digit(index_to_show)

# # affichage des statistiques (nbr d'occurrences des chiffres de 0 à 9)
# mnist_loader.display_basic_statistics()

# # affichage du chiffre donnée
# mnist_loader.display_image(0, dataset='test')

# # affichage dde la moyenne des chiffres 
# mnist_loader.display_mean_digits()

# mnist_loader.reshape_images()
