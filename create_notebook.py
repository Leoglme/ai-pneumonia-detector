import nbformat as nbf
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour charger les résultats
def load_results(file_path):
    with open(file_path, "r") as file:
        return eval(file.read())

# Création d'un nouveau notebook
n = nbf.v4.new_notebook()

# Titre
n.cells.append(nbf.v4.new_markdown_cell('## Analyse des résultats des modèles KNN et CNN'))

# Chargement des résultats
knn_results = load_results("knn_results.txt")
cnn_results = load_results("cnn_results.txt")

# Ajouter les résultats KNN
n.cells.append(nbf.v4.new_markdown_cell('### Résultats du modèle KNN'))
knn_code = f"""
knn_results = {knn_results}
print("Précision KNN : ", knn_results['accuracy'])
print("Score F1 KNN : ", knn_results['f1_score'])
print("Rapport de classification KNN :\\n", knn_results['classification_report'])
"""
n.cells.append(nbf.v4.new_code_cell(knn_code))

# Ajouter les résultats CNN
n.cells.append(nbf.v4.new_markdown_cell('### Résultats du modèle CNN'))
cnn_code = f"""
cnn_results = {cnn_results}
print("Précision CNN : ", cnn_results['accuracy'])
print("Score F1 CNN : ", cnn_results['f1_score'])
print("Rapport de classification CNN :\\n", cnn_results['classification_report'])
"""
n.cells.append(nbf.v4.new_code_cell(cnn_code))

# Comparaison des modèles
n.cells.append(nbf.v4.new_markdown_cell('### Comparaison des modèles'))
comparison_code = """
import matplotlib.pyplot as plt

# Données pour la comparaison
labels = ['KNN', 'CNN']
accuracy = [knn_results['accuracy'], cnn_results['accuracy']]
f1_scores = [knn_results['f1_score'], cnn_results['f1_score']]

# Création des graphiques
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Précision
axs[0].bar(labels, accuracy, color=['blue', 'green'])
axs[0].set_title('Comparaison de la précision')
axs[0].set_ylabel('Précision')

# Score F1
axs[1].bar(labels, f1_scores, color=['blue', 'green'])
axs[1].set_title('Comparaison du score F1')
axs[1].set_ylabel('Score F1')

plt.tight_layout()
plt.show()
"""
n.cells.append(nbf.v4.new_code_cell(comparison_code))

# Conclusion
n.cells.append(nbf.v4.new_markdown_cell("""
## Conclusion
En comparant les deux modèles, le CNN semble être plus performant pour notre cas d'utilisation en raison de sa précision et de son score F1 supérieurs. Cette performance peut être attribuée à la capacité du CNN à mieux capter les caractéristiques spatiales dans les images de radiographie pulmonaire.
"""))

# Enregistrement du notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
