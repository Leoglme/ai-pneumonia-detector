import nbformat as nbf
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_results(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Load KNN results
knn_results = load_results("knn_results.json")
knn_accuracy = knn_results["accuracy"]
knn_report = knn_results["classification_report"]
knn_cm = knn_results["confusion_matrix"]

# Load CNN results
cnn_results = load_results("cnn_results.json")
cnn_accuracy = cnn_results["test_accuracy"]

# Create a new notebook
n = nbf.v4.new_notebook()

# Add a text cell for titles
n.cells.append(nbf.v4.new_markdown_cell('## Résultats des modèles KNN et CNN'))

# Add KNN results with explanations
n.cells.append(nbf.v4.new_markdown_cell('### Résultats du modèle KNN'))
n.cells.append(nbf.v4.new_code_cell(f"""
print("Précision du KNN: {knn_accuracy * 100:.2f}%")
print(\"\"\"
Rapport de classification:
              precision    recall  f1-score   support

      NORMAL       0.96      0.77      0.86       324
   PNEUMONIA       0.92      0.99      0.95       847

    accuracy                           0.93      1171
   macro avg       0.94      0.88      0.90      1171
weighted avg       0.93      0.93      0.93      1171
\"\"\")
print("Matrice de confusion: \\n", knn_cm)
"""))

# Add CNN results with explanations
n.cells.append(nbf.v4.new_markdown_cell('### Résultats du modèle CNN'))
n.cells.append(nbf.v4.new_code_cell(f"""
print("Précision du CNN: {cnn_accuracy * 100:.2f}%")
"""))

# Add a comparison of the models
n.cells.append(nbf.v4.new_markdown_cell('### Comparaison des modèles KNN et CNN'))
n.cells.append(nbf.v4.new_code_cell(f"""
knn_vs_cnn = {{
    'Modèle': ['KNN', 'CNN'],
    'Précision': [{knn_accuracy}, {cnn_accuracy}]
}}

df_comparison = pd.DataFrame(knn_vs_cnn)
print(df_comparison)
"""))

# Plotting comparison
n.cells.append(nbf.v4.new_markdown_cell('### Visualisation de la comparaison des précisions'))
n.cells.append(nbf.v4.new_code_cell("""
import matplotlib.pyplot as plt

df_comparison.plot(kind='bar', x='Modèle', y='Précision', legend=False)
plt.title('Comparaison des précisions entre KNN et CNN')
plt.xlabel('Modèle')
plt.ylabel('Précision')
plt.ylim(0, 1)
plt.show()
"""))

# Add a conclusion
n.cells.append(nbf.v4.new_markdown_cell('### Conclusion'))
n.cells.append(nbf.v4.new_markdown_cell(\"\"\"
Après avoir comparé les deux modèles, nous constatons que le modèle CNN est plus précis que le modèle KNN pour la détection de la pneumonie. 
La CNN est plus adaptée pour cette tâche car elle peut capturer des caractéristiques plus complexes dans les images médicales grâce à ses couches convolutives.
\"\"\"))

# Save the notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
