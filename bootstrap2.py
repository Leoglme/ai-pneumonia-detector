import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # bibliothèque de visualisation de données en Python.
from keras.datasets import mnist

class MNISTLoader:
    def __init__(self):
        # Chargement des données MNIST
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.preprocess_data()
        self.reshape_images()
    
    def preprocess_data(self):
        # Normalisation des données pour les mettre à l'échelle entre 0 et 1
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.

    def reshape_images(self):
        # Remodeler les images pour les adapter à la taille [n, 784]
        self.x_train = self.x_train.reshape(-1, 28*28)
        self.x_test = self.x_test.reshape(-1, 28*28)

    def train_and_evaluate(self):
        # Création du modèle KNN
        knn = KNeighborsClassifier(n_neighbors=3)  # Choix de k = 3
        # Chaque prédiction sera basée sur les 3 voisins les plus proches.

        # Entraînement du modèle
        knn.fit(self.x_train, self.y_train)

        # Prédiction sur l'ensemble de test
        y_pred = knn.predict(self.x_test)

        # Évaluation de la précision
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Précision du modèle KNN: {accuracy * 100:.2f}%")
        
    # Appel des fonctions de visualisation
        visualize_misclassifications(self, y_pred)
        
def visualize_misclassifications(self, y_pred):
    # Calcul du pourcentage d'exemples mal classés pour chaque chiffre
        misclassified_indices = np.where(self.y_test != y_pred)[0]
        misclassifications = {i: 0 for i in range(10)}
        for idx in misclassified_indices:
            true_label = self.y_test[idx]
            misclassifications[true_label] += 1

        total_per_class = {i: sum(self.y_test == i) for i in range(10)}
        misclassification_percentages = {i: (misclassifications[i] / total_per_class[i]) * 100 for i in range(10)}

        # Affichage des pourcentages
        print("Pourcentage d'exemples mal classés par chiffre :")
        for digit, percentage in misclassification_percentages.items():
            print(f"{digit}: {percentage:.2f}%")

        # Représentation graphique
        plt.bar(misclassification_percentages.keys(), misclassification_percentages.values())
        plt.xlabel('Chiffres')
        plt.ylabel('Pourcentage de mauvaise classification')
        plt.title('Pourcentage d\'exemples mal classés par chiffre')
        plt.show()

        # Affichage des mauvaises classifications pour chaque chiffre
        for digit in range(10):
            print(f"Mauvaises classifications pour le chiffre {digit}:")
            misclassified_indices_for_digit = misclassified_indices[self.y_test[misclassified_indices] == digit]
            for idx in misclassified_indices_for_digit[:5]:  # Limiter à 5 exemples par chiffre
                plt.imshow(self.x_test[idx].reshape(28, 28), cmap='gray')
                plt.title(f"Vrai: {self.y_test[idx]}, Prédit: {self.y_pred[idx]}")
                plt.show()        



# Utilisation de la classe et évaluation du modèle
mnist_loader = MNISTLoader()
mnist_loader.train_and_evaluate()
