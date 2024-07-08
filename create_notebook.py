import nbformat as nbf

n = nbf.v4.new_notebook()

# Introduction avec un style personnalisé
n.cells.append(nbf.v4.new_markdown_cell("""
## Introduction

Ce projet vise à comparer deux approches de machine learning - les réseaux de neurones convolutifs (CNN) et les K-Nearest Neighbors (KNN) - pour détecter la pneumonie à partir d'images de radiographie. Nous évaluerons ces modèles en fonction de leur précision, de leur perte, et d'autres métriques pertinentes, incluant les faux négatifs, faux positifs, et des graphiques tels que la matrice de confusion et la courbe ROC.

<style>
h1 {color: navy;}
</style>
"""))

# Structure du Dataset
n.cells.append(nbf.v4.new_markdown_cell("""
### Structure du Dataset

Le dataset se compose de trois dossiers principaux :
- **train** : utilisé pour l'entraînement des modèles
- **val** : utilisé pour la validation des modèles après l'entraînement
- **test** : utilisé pour tester les modèles

Chaque dossier contient deux sous-dossiers :
- **NORMAL** : contient des images de radiographies normales
- **PNEUMONIA** : contient des images de radiographies avec pneumonie
"""))

# Explication de la classe ImageUtils
n.cells.append(nbf.v4.new_markdown_cell("""
### Filtrage des images avec ImageUtils

La classe `ImageUtils` est utilisée pour filtrer les images de radiographie afin de s'assurer qu'elles respectent certaines dimensions minimales. Voici comment elle fonctionne :

```python
class ImageUtils:
    @staticmethod
    def filter_images(data_dir, img_size=(256, 256)):
        \"""
        Filter out images that are too small and gather statistics on the image sizes.
        \"""
        min_img_size = (img_size[0] * 2, img_size[1] * 2)
        image_stats = {
            'total_images': 0,
            'min_size': (float('inf'), float('inf')),
            'max_size': (0, 0),
            'avg_width': 0,
            'avg_height': 0,
            'filtered_images': 0
        }
        filepaths = []
        total_width, total_height = 0, 0

        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    filepath = os.path.join(dirpath, filename)
                    with Image.open(filepath) as img:
                        width, height = img.size
                        image_stats['total_images'] += 1
                        total_width += width
                        total_height += height
                        if width < min_img_size[0] or height < min_img_size[1]:
                            image_stats['filtered_images'] += 1
                        else:
                            filepaths.append(filepath)
                            image_stats['min_size'] = (
                                min(image_stats['min_size'][0], width),
                                min(image_stats['min_size'][1], height)
                            )
                            image_stats['max_size'] = (
                                max(image_stats['max_size'][0], width),
                                max(image_stats['max_size'][1], height)
                            )
                            image_stats['avg_width'] += width
                            image_stats['avg_height'] += height

        remaining_images = image_stats['total_images'] - image_stats['filtered_images']
        if remaining_images > 0:
            image_stats['avg_width'] /= remaining_images
            image_stats['avg_height'] /= remaining_images

        image_stats['avg_width'] = round(image_stats['avg_width'])
        image_stats['avg_height'] = round(image_stats['avg_height'])

        filtered_filepaths = [fp for fp in filepaths if
                              Image.open(fp).size[0] >= min_img_size[0] and Image.open(fp).size[1] >=
                              min_img_size[1]]
        return filtered_filepaths, image_stats
```
"""))

# Ajout du code pour charger les résultats, afficher les détails, et générer des graphiques
code = """
import json
import matplotlib.pyplot as plt
import numpy as np

# Chargement des résultats
with open("knn_results.json", "r") as file:
    knn_results = json.load(file)
with open("cnn_results.json", "r") as file:
    cnn_results = json.load(file)

# Affichage des résultats KNN
print("## Résultats du modèle KNN\\n- Précision: {}".format(knn_results['accuracy']))
print("- Rapport de classification:\\n{}".format(knn_results['classification_report']))

# Affichage des résultats CNN
print("\\n## Résultats du modèle CNN\\n- Précision de test: {}".format(cnn_results['test_accuracy']))
print("- Perte de test: {}".format(cnn_results['test_loss']))

# Données pour la comparaison
labels = ['KNN', 'CNN']
accuracy = [knn_results['accuracy'], cnn_results['test_accuracy']]

# Création des graphiques
plt.figure(figsize=(8, 6))
plt.bar(labels, accuracy, color=['blue', 'green'])
plt.title('Comparaison de la précision')
plt.ylabel('Précision')
plt.show()
"""
n.cells.append(nbf.v4.new_code_cell(code))

# Explication du modèle KNN
n.cells.append(nbf.v4.new_markdown_cell("""
### Modèle KNN

Le modèle KNN (K-Nearest Neighbors) est un modèle de machine learning simple mais efficace pour la classification. Voici comment il est construit et utilisé dans ce projet :

1. **Prétraitement des images** : Les images sont redimensionnées et normalisées.
2. **Création du dataset** : Les images sont transformées en vecteurs et divisées en ensembles d'entraînement et de validation.
3. **Entraînement** : Le modèle KNN est entraîné sur les images d'entraînement.
4. **Évaluation** : Le modèle est évalué sur les images de validation pour calculer la précision et d'autres métriques.
"""))

# Graphique KNN
n.cells.append(nbf.v4.new_markdown_cell("""
# Graphique KNN
<style>
h1 {color: navy;}
</style>
"""))

code2 = """
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Chargement des résultats
with open("knn_results.json", "r") as file:
    knn_results = json.load(file)

# Convertir la matrice de confusion en tableau NumPy
cm = np.array(knn_results['confusion_matrix'])

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC curve
plot_roc_curve(knn_results['fpr'], knn_results['tpr'], knn_results['roc_auc'])

# Plot Confusion Matrix
plot_confusion_matrix(cm)
"""
n.cells.append(nbf.v4.new_code_cell(code2))

# Explication du modèle CNN
n.cells.append(nbf.v4.new_markdown_cell("""
### Modèle CNN

Le modèle CNN (Convolutional Neural Network) est un modèle de machine learning puissant pour la classification d'images. Voici comment il est construit et utilisé dans ce projet :

1. **Prétraitement des images** : Les images sont redimensionnées et normalisées.
2. **Création du dataset** : Les images sont transformées en vecteurs et divisées en ensembles d'entraînement et de validation.
3. **Entraînement** : Le modèle CNN est construit avec plusieurs couches de convolution, de pooling et de couches denses.
4. **Évaluation** : Le modèle est évalué sur les images de validation pour calculer la précision et d'autres métriques.
"""))

# Graphique CNN
n.cells.append(nbf.v4.new_markdown_cell("""
# Graphique CNN
<style>
h1 {color: navy;}
</style>
"""))

code3 = """
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Chargement des résultats
with open("cnn_results.json", "r") as file:
    cnn_results = json.load(file)

# Convertir la matrice de confusion en tableau NumPy
cm = np.array(cnn_results['confusion_matrix'])

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1

], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC curve
plot_roc_curve(cnn_results['fpr'], cnn_results['tpr'], cnn_results['roc_auc'])

# Plot Confusion Matrix
plot_confusion_matrix(cm)
"""
n.cells.append(nbf.v4.new_code_cell(code3))

# Conclusion
n.cells.append(nbf.v4.new_markdown_cell("""
## Conclusion

En comparant les deux modèles, nous observons que les deux approches ont des performances solides pour la détection de la pneumonie à partir d'images de radiographie. 

Le modèle CNN montre une meilleure précision et un score AUC ROC plus élevé, ce qui indique une meilleure capacité à distinguer entre les cas de pneumonie et les cas normaux. Cependant, le modèle KNN reste une option valide et plus simple à implémenter.

Pour des applications nécessitant une haute précision et la capacité de traiter de grands ensembles de données avec des images complexes, le modèle CNN est recommandé. En revanche, pour des scénarios où les ressources informatiques sont limitées et une implémentation rapide est nécessaire, le modèle KNN peut être plus approprié.
"""))

# Enregistrement du notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)