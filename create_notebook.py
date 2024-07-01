import nbformat as nbf
import json
import matplotlib.pyplot as plt

# Fonction pour charger les résultats depuis un fichier JSON
def load_results(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Création d'un nouveau notebook
n = nbf.v4.new_notebook()

# Ajouter le titre
n.cells.append(nbf.v4.new_markdown_cell('## Analyse des résultats des modèles KNN et CNN'))

# Chargement des résultats
knn_results = load_results("knn_results.json")
cnn_results = load_results("cnn_results.json")

# Ajouter les résultats du modèle KNN
knn_text = f"""
### Résultats du modèle KNN
- Précision: {knn_results['accuracy']}
- Rapport de classification:\n{knn_results['classification_report']}
"""
n.cells.append(nbf.v4.new_markdown_cell(knn_text))

# Ajouter les résultats du modèle CNN
cnn_text = f"""
### Résultats du modèle CNN
- Précision de test: {cnn_results['test_accuracy']}
- Perte de test: {cnn_results['test_loss']}
"""
n.cells.append(nbf.v4.new_markdown_cell(cnn_text))

# Ajouter des graphiques de comparaison
n.cells.append(nbf.v4.new_code_cell("""
%matplotlib inline
import matplotlib.pyplot as plt

# Données pour la comparaison
labels = ['KNN', 'CNN']
accuracy = [knn_results['accuracy'], cnn_results['test_accuracy']]

# Création des graphiques
fig, ax = plt.subplots(figsize=(8, 6))

# Graphique de précision
ax.bar(labels, accuracy, color=['blue', 'green'])
ax.set_title('Comparaison de la précision')
ax.set_ylabel('Précision')

plt.tight_layout()
plt.show()
"""))

# Conclusion
n.cells.append(nbf.v4.new_markdown_cell("""
## Conclusion
En comparant les deux modèles, nous observons des différences en termes de précision. Il est important de considérer les caractéristiques spécifiques des données et les exigences du problème pour choisir le modèle approprié.
"""))

# Enregistrement du notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
