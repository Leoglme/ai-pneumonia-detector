import nbformat as nbf
import matplotlib.pyplot as plt
import numpy as np

# Create a new notebook
n = nbf.v4.new_notebook()

# Add a text cell for titles
n.cells.append(nbf.v4.new_markdown_cell('## Results from KNN and CNN Models'))

# Add code cell for KNN results
knn_code = """
knn_results = open("knn_results.txt", "r").read()
print(knn_results)
"""
n.cells.append(nbf.v4.new_code_cell(knn_code))

# Add code cell for CNN results
cnn_code = """
cnn_results = open("cnn_results.txt", "r").read()
print(cnn_results)
"""
n.cells.append(nbf.v4.new_code_cell(cnn_code))

# Save the notebook
with open('ai_model_notebook.ipynb', 'w') as f:
    nbf.write(n, f)
