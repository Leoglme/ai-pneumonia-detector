import nbformat as nbf

# Création d'un nouveau notebook
n = nbf.v4.new_notebook()

# Ajouter le titre et les résultats dans une seule cellule de code pour simplifier l'exécution
code = """
import json
import matplotlib.pyplot as plt

# Titre
print("# Analyse des résultats des modèles KNN et CNN\\n")

# Chargement des résultats
with open("knn_results.json", "r") as file:
    knn_results = json.load(file)
with open("cnn_results.json", "r") as file:
    cnn_results = json.load(file)

# Affichage des résultats KNN
print("## Résultats du modèle KNN")
print("- Précision: {}".format(knn_results['accuracy']))
print("- Rapport de classification:\\n{}".format(knn_results['classification_report']))

# Affichage des résultats CNN
print("\\n## Résultats du modèle CNN")
print("- Précision de test: {}".format(cnn_results['test_accuracy']))
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

# Conclusion
print("\\n## Conclusion")
print("En comparant les deux modèles, nous observons des différences en termes de précision. Il est important de considérer les caractéristiques spécifiques des données et les exigences du problème pour choisir le modèle approprié.")
"""

# Ajout de la cellule de code
n.cells.append(nbf.v4.new_code_cell(code))

# Enregistrement du notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
