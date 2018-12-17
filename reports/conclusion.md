## Conclusión

En este trabajo presentamos un algoritmo para visualizar datos en dimensiones muy altas, así como sus extensiones para ser más rápido, utilizar datos etiquetados y extenderse a puntos fuera del conjunto de entrenamiento. Mostramos también un caso de uso en MNIST, un conjunto en 784 dimensiones, que el algoritmo separa casi perfectamente. 

Aunque el análisis de convergencia relaciona t-SNE con métodos de clustering, seguimos a van der Maaten en sugerir que se utilize exclusivamente para visualización en 2 y 3 dimensiones. Para reducción de dimensionalidad en casos más generales, sugerimos utilizar UMAP, un algoritmo más poderoso y más general.