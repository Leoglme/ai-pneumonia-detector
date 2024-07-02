import nbformat as nbf

n = nbf.v4.new_notebook()

# Ajouter un titre avec un style personnalisé
n.cells.append(nbf.v4.new_markdown_cell("""
# Analyse des résultats des modèles KNN et CNN
<style>
h1 {color: navy;}
</style>
"""))

# Code pour charger les résultats, afficher les détails, et générer des graphiques
code = """
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

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

def plot_roc_curve(self, fpr, tpr, roc_auc):
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

def plot_confusion_matrix(self, cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Plot ROC curve
plot_roc_curve(knn_results['fpr'], knn_results['tpr'], knn_results['roc_auc'])

# Plot Confusion Matrix
plot_confusion_matrix(knn_results['confusion_matrix'])
"""

n.cells.append(nbf.v4.new_code_cell(code))

# Conclusion avec mise en forme
n.cells.append(nbf.v4.new_markdown_cell("""
## Conclusion
**En comparant les deux modèles, nous observons des différences en termes de précision. Il est important de considérer les caractéristiques spécifiques des données et les exigences du problème pour choisir le modèle approprié.**
"""))

# Enregistrement du notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
