# Propuesta de proyecto: Métodos no lineales de reducción de dimensión.

## Introducción

Durante el curso se estudió la técnica central para reducción de dimensión: el método de componentes principales. Pese a sus diversas aplicaciones contiene una limitación intrínseca, es incapaz de capturar relaciones no lineales entre los datos. Como consecuencia surgen una gran variedad de técnicas más generales que pretenden resolver este problema. 

Los métodos no lineales de reducción de dimensión abordan el problema general realizando una simplificación al suponer que la estructura subyacente de los datos de interés es una *variedad* inmersa dentro del espacio original de estos.

## Un enfoque topológico

Técnicamente,  un espacio topológico $\mathcal{M}$ Hausdorff, segundo numerable y localmente homeomorfo al espacio euclidiano $\mathbb{R}^n$ se llama *variedad topológica de dimensión n*. 

Para los propósitos de este proyecto, trabajaremos exclusivamente con *variedades diferenciables*. De manera intuitiva, una variedad diferencial es una superficie suave o curva inmersa en el espacio euclidiano.

Una observación importante es que una variedad $q$-dimensional puede ser localmente aproximada por un subespacio lineal de dimensión $q$. Dicho subespacio es conocido como el *subespacio tangente* a la variedad.  La mayoría de los métodos de reducción de dimensión (ISOMAP,  LLE,  *Laplacian Eigenmaps*) explotan esta propiedad para preservar la estructura local de los datos en cuestión.

## Proyecto

En este proyecto exploraremos el marco teórico que sustenta algunas de las técnicas no lineales de reducción de dimensionalidad más utilizadas, entre las cuales se encuentran:

- Isometric Mapping (ISOMAP)
- Locar Linear Embedding (LLE)
- Laplacian Eigenmap
- Self Organizing Map 
- t-Distributed Stochastic Neighbor Embedding (t-SNE)

Presentaremos los algoritmos correspondientes y realizaremos un estudio comparativo sobre el desempeño de dichos métodos en uno (o varios) conjuntos de datos. Asimismo, analizaremos las aplicaciones de las técnicas no lineales de reducción de dimensión en comparación con su contraparte lineal enfatizando en cada uno un caso de uso y una discusión sobre interpretabilidad.

## Referencias

[1] Van der Maaten, L., & Geoffrey, H. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 2579–2605. Retrieved from https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf

[2] Harmeling, S. (2007). Exploring model selection techniques for nonlinear dimensionality reduction. *Neural Computation*.

[3] Tenenbaum, J. B., de Silva, V., & Langford, J. C. (1995). *A Global Geometric Framework for Nonlinear Dimensionality Reduction*. *Philos. Trans. R. Soc. London Ser. B* (Vol. 67). Retrieved from www.sciencemag.org

[4] https://www.stat.cmu.edu/~cshalizi/350/lectures/14/lecture-14.pdf

[5] http://www.matmor.unam.mx/~ferran/GeoDif.pdf