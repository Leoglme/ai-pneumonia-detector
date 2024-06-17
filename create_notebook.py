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

# Code pour afficher les résultats KNN
knn_code = f"""
print("Résultats KNN :\\n{knn_results}")
"""
n.cells.append(nbf.v4.new_code_cell(knn_code))

# Code pour afficher les résultats CNN
cnn_code = f"""
print("Résultats CNN :\\n{cnn_results}")
"""
n.cells.append(nbf.v4.new_code_cell(cnn_code))

# Comparaison des modèles
comparison_code = """
# Code pour générer des graphiques comparatifs
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].bar(['KNN', 'CNN'], [knn_results['accuracy'], cnn_results['accuracy']])
axs[0].set_title('Comparaison de la précision')
axs[0].set_ylabel('Précision')

axs[1].bar(['KNN', 'CNN'], [knn_results['f1_score'], cnn_results['f1_score']])
axs[1].set_title('Comparaison du score F1')
axs[1].set_ylabel('Score F1')
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
