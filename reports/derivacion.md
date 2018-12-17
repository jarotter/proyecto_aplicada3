

## t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE es una técnica de visualización desarrollada en 2008 por Laurens Van der Maaten y Geoffrey Hinton que consiste en visualizar datos en altas dimensiones al asignar a cada punto una localización en el espacio euclidiano de 2 o 3 dimensiones. Dicha técnica surge como una mejora a SNE (Stochastic Neighbor Embedding) al tener un gradiente más sencillo de optimizar y que resuelve el *problema de aglutinamiento*. A continuación presentamos t-SNE y la optimización de Barnes-Hut propuesta por el mismo Van der Maaten en el 2014. Esta técnica permite la escalabilidad del algoritmo a conjuntos de datos más grandes. Actualmente t-SNE es considerada como una de las técnicas de *estado del arte* para visualización y reducción de dimensión. 

#### Stochastic Neighbor Embedding (SNE)

Al igual que en escalamiento multidimensional, en SNE definimos una medida de similaridad. A saber, el vecino más cercano de $\bold{x_i}$ es el punto $\bold{x_j}$ con mayor probabilidad bajo una distribución normal  centrada en $\mathbf{x}_i$ con varianza $\sigma_i^2\mathbb{I} $. Como la varianza depende de $i$, esta similaridad es asimétrica. Formalmente
$$
\begin{align}
p_{j\vert i} &= \frac{\exp\left(-\Vert \bold{x_i} - \bold{x_j} \Vert^2 / 2\sigma_i^2 \right)}{\sum\limits_{k\not=i}\exp\left(-\Vert \bold{x_i} - \bold{x_k} \Vert^2 / 2\sigma_i^2\right)}
\end{align}
$$
Y $p_{i\vert i} = 0$

La varianza $\sigma_i^2$ se calcula determinísticamente a partir de un hiperparámetro especificado por el usuario y de manera que tome valores más pequeños si $\mathbf{x}_i$ se encuentra en una zona de poca concentración $\sigma_i$ y valores más grandes si $\mathbf{x}_i$ se encuentra en una zona de mayor concentración.

Sea $P_i$ la distribución de probabilidad condicional sobre las demás observaciones dada $\mathbf{x}_i$, *la entropía de Shanon* de $P_i$ (para más detalles ver apéndice A) es
$$
H(P_i) := - \sum_j p_{j|i} \log p_{j|i}
$$

Notemos que cualquier valor de la varianza $\sigma_i^2$ induce una distribución de probabilidad $P_i$ cuya entropía $H(P_i)$ es creciente en $\sigma^2_i$. En consecuencia definimos la *perplexity* de $P_i$ como

$$
\mathrm{Perp}(P_i) = 2^{H(P_i)}
$$

Que puede ser interpretada como una medida continua de el número efectivo de vecinos. Así, SNE ejecuta una búsqueda binaria para obtener el valor de $\sigma_i$ que produce una distribución de probabilidad $P_i$ con un valor *perplexity* especificado de antemano por el usuario.

De manera análoga a la construcción de $p_{j|i}$ podemos obtener la densidad condicional $q_{j|i}$ de las $\mathbf{y}_i$ por medio de la siguiente expresión
$$
q_{j\vert i} = \frac{\exp\left(-\Vert \bold{y_i} - \bold{y_j} \Vert^2 \right)}{\sum\limits_{k\not=i}\exp\left(-\Vert \bold{y_i} - \bold{y_k} \Vert^2 \right)}
$$
En constraste con $(1)$ fijamos la varianza de las distribuciones normales correspondientes en $\frac{1}{\sqrt{2}}$.

En caso de que los puntos mapeados $\mathbf{y}_i$ y $\mathbf{y}_j$ modelen de forma correcta la similaridad entre las observaciones $\mathbf{x}_i$ y $\mathbf{x}_j$ debería tenerse que $p_{j|i} = q_{j|i}$. Con base en esta observación, SNE propone minimizar la divergencia de Kullback-Leibler entre las distribuciones condicionales $P_i$ y $Q_i$ (para mayor intuición sobre porque la divergencia de KL surge de manera natural en este contexto, visitar el apéndice A). La función costo C está dada por
$$
C = \sum _i D_{KL} (P_i \Vert Q_i) = \sum_i \sum_j p_{i|j}\log\frac{p_{j|i}}{q_{j|i}}
$$


Para minimizar $C$ utilizaremos el método de descenso por gradiente, a saber 
$$
\nabla_{y_i} C = 2 \sum_j (p_{j\vert i} + p_{i \vert j} - q_{j \vert i } - q_{i \vert j})(y_i - y_j)
$$

#### Construcción de t-SNE

SNE presenta dos problemas que reducen considerablemente su uso práctico.

1. La función costo dificil de optimizar.
2. Se presenta el *problema de aglutinamiento* sobre el cual se mencionará más a continuación.

t-SNE surge como alternativa para solucionar estos conflictos.

###### Simetría

Una alternativa (más sencilla) para la función costo surge al añadir simetría a las funciones involucradas en la divergencia de KL. En este caso, en vez de minimizar la suma de las divergencias de KL entre las condicionales $p_{i \vert j}$ y $q_{i \vert j}$ utilizaremos las distribuciones conjuntas 
$$
q_{i j} = \frac{\exp\left(-\Vert \bold{y_i} - \bold{y_j} \Vert^2 \right)}{\sum\limits_{k}\sum\limits_{l\not=k}\exp\left(-\Vert \bold{y_i} - \bold{y_k} \Vert^2 \right)}\nonumber \\ 

p_{i j} = \frac{\exp\left(-\Vert \bold{y_i} - \bold{y_j} \Vert^2 \right)}{\sum\limits_{k}\sum\limits_{l\not=k}\exp\left(-\Vert \bold{y_i} - \bold{y_k} \Vert^2 \right)}
$$
Notemos que si $\mathbf{x}_i$ es un *outlier*, $p_{ij}$ es pequeño para toda $j$, por lo que la influencia de su correspondiente $\mathbf{y}_i$ sobre la función objetivo es practicamente nula y $\mathbf{y}_i$ está poco relacionada con el resto. Esta situación se arregla definiendo la conjunta de $p$ de la siguiente forma 
$$
\begin{align}
p_{ij} &= \frac{p_{j\vert i }+p_{i\vert j}}{2n}  
\end{align}
$$
Notemos que $\sum_j p_{ij} > \frac{1}{2n} \ \forall i$ de forma que, aún si $\mathbf{x}_i$ es un outlier la influencia sobre la función objetivo de su $\mathbf{y}_i$ es significativa. Así pues, si $P$ y $Q$  son las distribuciones conjuntas la función costo es
$$
C = \sum _i D_{KL} (P \Vert Q) = \sum_i \sum_j p_{ij}\log\frac{p_{ij}}{q_{ij}}
$$
Cuyo gradiente presenta una forma más sencilla que su contraparte asimétrica. 
$$
\nabla_{y_i} = 4 \sum_{j} (p_{ij} - q_{ij})(y_i - y_j)
$$


###### Problema de aglutinamiento

El problema de aglutinamiento es una consecuencia de la *maldición de la dimensionalidad*, la forma más sencilla de presentarlo para el caso particular que estamos enfrentando es la siguiente. Supongamos que el espacio original está en $n$ dimensiones y se quiere realizar un mapeo a un espacio de dimensión $p \ll n$., notemos que si hay $n+1$ puntos $\mathbf{x}_1, ..., \mathbf{x}_{n+1}$ equidistantes entre sí no hay forma de preservar esta propiedad en dimensiones menores a $n$.

En general, dado que el mapeo propuesto por SNE prioriza la preservación de distancias pequeñas, muchos de los puntos que se encuentran a distancias moderadas de $\mathbf{x}_i$ serán mapeados a puntos que se encuentran distantes de $\mathbf{y}_i$ pues, como se mencionó en la introducción, una consecuencia de la maldición de la dimensionalidad es que en el espacio bajo-dimensional *tenemos menos espacio*. Todos estos puntos ejercen una fuerza atractiva (pequeña pero considerable) hacia $\mathbf{y}_i$ que, si se acumula, termina por eliminar los espacios que inicialmente se habían formado entre los clusters naturales de los datos.

###### Distribuciones de colas pesadas como solución al desajuste entre distancias

Para resolver el problema de aglutinamiento, t-SNE propone utilizar una distribución de colas pesadas  en $\mathbb{R}^q$ de forma que las fuerzas atracticas ocasionadas por la obtención de grandes distancias entre puntos separados moderadamente en el espacio original, puedan ser modeladas en la cola de la distribución.

En este caso se utiliza una distribución $t$ con un grado de libertad (una Cauchy)de forma que
$$
q_{ij} = \frac{(1 + \Vert y_i - y_j \Vert)^{-1}}{\sum\limits_k\sum\limits_{l\not = k}(1 + \Vert y_k - y_l \Vert^2)^{-1}}
$$
Con dicho cambio, el gradiente de la nueva función costo está dado por
$$
\nabla_{y_i}C = 4\sum_j (p_{ij} - q_{ij})(y_i - y_j) (1 + \Vert y_i - y_j \Vert^2)^{-1}
$$

Las fuerzas repulsivas en el gradiente de t-SNE son mucho mayores que en el gradiente de SNE obteniendo un proceso más equilibrado (a diferencia del proceso predominantemente atractivo que se genera en SNE). Para un análisis más detallado basado en mecánica de partículas ver [6].

El proceso de descenso por gradiente es inicializado en $\mathcal{Y}^0$, una muestra de una distribución normal con media cero y varianza pequeña. Para acelerar el descenso y para evitar caer en mínimos locales se añade una tasa de aprendizaje, $\eta$  y un *momentum* $\alpha$ al gradiente. Esta técnica preserva cierta memoria (que decae exponencialmente) de los gradientes anteriores. El cálculo final es
$$
\mathcal{Y^{(t)}} = \mathcal{Y^{(t-1)}} + \eta \nabla_\mathcal{Y}C + \alpha(t)\left(\mathcal{Y^{(t-1)}} - \mathcal{Y^{(t-2)}}\right)
$$
Los autores recomiendan $\eta=100$ y actualizaciones en cada iteración con el esquema descrito en [7] para actualizar $\eta$. Para $\alpha$, usamos  
$$
\begin{cases}
\alpha^{(t)} = .5 \quad \text{si} \ t<250 \\ \nonumber
\alpha^{(t)} = .8 \quad \text{si} \ t\geq250 \\
\end{cases}
$$

###### Algoritmo: t-SNE

```pseudocode
t-SNE(data=x, perp=30, T=1000, alpha=alpha(t) nu=100):
""" t-SNE: t-Distributed Stochastic Neighbor Embedding.
	
	Parámetros:
	-----------
	data (data frame):
		Las observaciones {x_1, ... x_n}.
	perp (double):
		Perplexity.
	T (int):
		Número de iteraciones
	nu (real):
		Tasa de apendizaje
	alpha (function):
		Momentum
	Regresa:
	--------
	Y (arreglo):
		Mapeo en bajas dimensiones de los datos originales
"""

	p(i|j) <- Calcula (1) utilizando la perp para obtener las varianzas 
	p(i,j) <- Calcula (7)
	Y <- Obten una muestra de una distribución con media cero y varianza 1e-4
    
	for t=1 hasta T:
		q(i,j) <- Calcula (10)
		grad_yi(C) <- Obtén (11)
		Y <- Calcula la actualización (14)
		
	return (Y)
```

###### Comentarios sobre perplexity

Como mencionamos, la *perplexity* puede ser interpretada como el número efectivo de vecinos. Valores muy pequeños de esta harán que el algoritmo se concentre únicamente en la estructura local y pierda completamente la noción de estructura global, mientras que para valores muy altos, la inmersión es deficiente y en ciertas ocasiones no se logra la convergencia.

A continuación se presentan las visualizaciones obtenidas tras ejecutar t-SNE donde los puntos en el espacio original se encuentran distribuidos sobre un nudo trefoil en 3 dimensiones. 

| ![vp_trees_1](./img/perplexity_1.png) | ![vp_trees_1](./img/perplexity_2.png) | ![vp_trees_1](./img/perplexity_3.png) |
| :-----------------------------------: | :-----------------------------------: | :-----------------------------------: |
|               Perp = 2                |               Perp = 30               |              Perp = 100               |

###### Algunas optimizaciones para el algoritmo

- *Early compression:* La idea es forzar a que durante las primeras iteraciones los puntos mapeados se mantengan a distancias pequeñas entre sí, de esta forma es más fácil que los clusters se muevan dentro de todo el espacio correspondiente al mapeo y que se pueda encontrar una organización global para los datos. Esto se implementa añadiendo una penalización adicional a la función costo que es proporcional a la suma de las distancias al cuadrado entre los puntos mapeados y el origen durante las primeras iteraciones.

- *Early exaggeration:* Consiste en multiplicar todas las conjuntas $p_{ij}$ por algún factor fijo, digamos $\beta > 1$, tal que las conjuntas $q_{ij}$ sean pequeñas en comparación con su contraparte $p_{ij}$, de esta forma se obliga a que el proceso de optimización de una menor prioridad a las distancias en el mapeo en bajas dimensiones. En consecuencia hay más espacio libre para que los clusters se puedan mover por todo el espacio y para que los datos se organicen de una mejor manera globalmente.

#### Barnes-Hut-SNE

Barnes-Hut-SNE es una implementación computacional eficiente de t-SNE propuesta por van der Maaten en [8] que permite reducir la complejidad computacional de t-SNE de $O(N^2)$ a $O(N\log N)$.  Para ello comienza por utilizar una estructura de datos de árboles métricos para aproximar P y posteriormente aproxima los gradientes $\nabla_{y_i} C $ utilizando el algoritmo de Barnes-Hut.

###### Aproximando el cálculo de la similaridad

Si $\mathbf{x}_i$ y $\mathbf{x}_j$ son disimilares (lejanos) $p_{j \vert i}$ es muy cercano a cero, en consecuencia podemos realizar una aproximación para el cálculo de $p_{ij}$ definiendo $p_{j \vert i}$ de la siguiente manera
$$
p_{j \vert i} = \begin{cases} \frac{\exp\left(-\Vert \bold{x_i} - \bold{x_j} \Vert^2 / 2\sigma_i^2 \right)}{\sum\limits_{k\in\mathcal{N}_i}\exp\left(-\Vert \bold{x_i} - \bold{x_k} \Vert^2 / 2\sigma_i^2\right)} &\quad \text{Si } j \in \mathcal{N_i}  \\ 
0 &\quad \text{En otro caso}
\end{cases}
$$
Donde $\mathcal{N_i}$ denota los $\lfloor 3u \rfloor$ vecinos más cercanos a $x_i$ y $\sigma$ se obtiene de forma que la *perplexity* correspondiente sea $u$. Puesto que únicamente consideramos $\lfloor 3u \rfloor$ observaciones para calcular $p_{ij}$, es de esperar que se mejore sustancialmente el tiempo de ejecución del algoritmo. Sin embargo, para ello se supuso que conocemos los vecinos más cercanos para cada observación $\mathbf{x}_i$. Pese a que ese no es el caso, existe una estructura de datos que nos permite obtener los $k$ vecinos más cercanos a $\mathbf{x}_i$ en $O(\log N)$, de forma que podemos encontrar el conjunto de vecinos más cercanos a todas las $\mathbf{x}_i$  en $O(N\log N )$. 

###### Árboles Vantage-Point (Árboles VP)

Los árboles VP son una estructura de datos métrica en la cual cada nodo almacena un objeto, en este caso un punto $\bold{x}_i$, y el radio de una bola con centro en dicho punto. El hijo izquierdo de cada nodo almacena todos aquellos puntos que se encuentren dentro de la bola y el hijo derecho almacena todos aquellos puntos que se encuentran fuera de ella . La construcción del árbol toma $O(N\log N)$ y las búsquedas para obtener los vecinos más cercanos se llevan a cabo en $O(\log N)$. A continuación se muestra visualmente el proceso de construcción.

| ![vp_trees_1](./img/vp_trees_1.png) | ![vp_trees_2](./img/vp_trees_2.png) | ![vp_trees_3](./img/vp_trees_3.png) |
| :---------------------------------: | :---------------------------------: | :---------------------------------: |
|          Primera iteración          |          Segunda iteración          |           Iteración final           |

######Aproximando el cálculo de el gradiente

Sea $Z := (1 + \Vert y_i - y_j \Vert^2)^{-1}$ y recordemos el gradiente de t-SNE para obtener
$$
\begin{align*}
\nabla_{y_i}C &= 4\sum_j (p_{ij} - q_{ij})(y_i - y_j) (1 + \Vert y_i - y_j \Vert^2)^{-1} \\
&= 4 \left( \sum_{j} p_{ij}(y_i - y_j)(1 + \Vert y_i - y_j \Vert^2)^{-1} - \sum_j q_{ij}(y_i-y_j)(1 + \Vert y_i - y_j \Vert^2)^{-1}\right)
\\
&= 4 \left( \sum_{j} p_{ij}q_{ij}(y_i - y_j)Z - \sum_j q_{ij}^2(y_i-y_j)Z\right)
\\
&= 4 (F_{a} - F_{r} )
\end{align*}
$$
$F_{a}$ denota la suma de todas las fuerzas *atractivas* y su cálculo ya es computacionalmente eficiente y puede obtenerse en $O(uN)$ tomando en cuenta $(13)$ y que $q_{ij}Z$ se puede calcular en $O(1)$.

$F_r$ denota la suma de fuerzas repulsivas y hasta el momento su cálculo presenta una complejidad de  $O(N^2)$, a continuación presentamos el algoritmo de Barnes-Hut que nos permitirá reducir la complejidad de la obtención de $F_r$ a $O(N\log N)$.

Considérense 3 puntos $\mathbf{y}_i, \mathbf{y}_j$ y $\mathbf{y}_k$ donde $\Vert \mathbf{y}_i - \mathbf{y}_j \Vert \approx \Vert \mathbf{y}_i - \mathbf{y}_k \Vert \gg \Vert \mathbf{y}_j - \mathbf{y}_k \Vert$. Entonces la diferencia entre las contribuciones de $\mathbf{y}_k$ y de $\mathbf{y}_j$ a $\sum_j q_{ij}^2(\mathbf{y}_i-\mathbf{y}_j)Z$ es prácticamente nula. El algoritmo de Barnes-Hut explota esta idea creando $m $ grupos ($g_1, ... g_n$) de puntos en $\{y_k\}_{k\not = i }$ tales que para cada grupo se cumplen las condiciones recién dadas.Podemos estimar $\sum_j q_{ij}^2(y_i-y_j)Z$ con
$$
\sum_{j=1}^m |g_j| \ q_{i,g_j} (y_i - y_{g_j})Z_{g_j}
$$
Donde $|g_j|$ es el número de elementos en el grupo $j$, y $q_{i,g_j}$ es la estimación de la densidad $q_{ij}$ en el grupo $j$ (usualmente se toma el centro de masa de los puntos en el grupo $j$ y con base en él se realiza la estimación). De la misma manera estimamos $y_{g_j}$ y $Z_{g_j}$.

Efectivamente, en caso de tener los grupos definidos la estimación reduce considerablemente el tiempo necesario para calcular la expresión correspondiente a las fuerzas repulsivas y por ende la complejidad del algoritmo en general ¿Cómo obtener los grupos y centros de masa correspondientes de manera eficiente? El autor propone utilizar una estructura de datos llamada *quadtree* para resolver dicho problema.

###### Quadtree 

*Quadtree* es una estructura de datos de tipo arbol en la cual cada nodo representa una celda rectangular con cierto centro, altura y anchura. Los nodos que no son hojas tienen 4 hijos que subdividen la celda en 4 cuadrantes. Asimismo para cada nodo se almacena el centro de masa de los puntos en esa celda y el número de elementos en ella. A continuación se muestra la división generada por el *quadtree* en el espacio original (lado izquierdo) y el arbol de búsqueda generado (lado derecho). 

Usando *quadtrees* podemos encontrar encontrar los grupos de arriba.

| ![vp_trees_1](./img/quadtree_1.png) | ![vp_trees_1](./img/quadtree_2.png) |
| ----------------------------------- | ----------------------------------- |
|                                     |                                     |

