## Extensiones

#### Multiple maps t-SNE

Las similaridades $  q_{ij}​$que se usan en t-SNE tienen una limitante que puede pasar desapercibida porque usualmente es algo que queremos: las propiedades de la métrica involucrada. Supongamos por ejemplo que los datos a visualizar son texto, y medimos la similaridad usando asociación entre palabras. Podría ser el caso, por ejemplo, que "lengua" tenga una alta similaridad a "tacos", pero también a "española". En este caso, t-SNE va a colocar a "española" y "tacos" más cerca de lo que en realidad deberían estar. Este efecto es inevitable por la construcción de las $q_{ij}​$, que utiliza la distancia euclidiana y obliga así, con la desigualdad del triángulo, a tener resultados transitivos.

Una extensión a t-SNE presentada por van der Maarten y Hinton en [3] construye $M$  *mapas*, realizaciones de t-SNE con todas las palabras que asigna a cada punto una importancia. Formalmente, la *importancia* del punto $\mathbf{x}_ i$ en el mapa $m$ es $\pi_i^{(m)}$ con las restricciones $\forall i \forall m\ \pi_i^{(m)} \geq 0$  y  $\sum_m\pi_i^{(m)}=1$. Las nuevas simiaridades en el espacio pequeño están dadas por
$$
q_{ij}\propto\sum_m\pi_i^{(m)}\pi_j^{(m)}\left(1+\left\|\mathbf{y}_i^{(m)}-\mathbf{y}_j^{(m)}\right\|^2\right)^{-1}
$$
con la constante de normalización apropiada para que sumen uno. Al optimizar la divergencia de Kullback-Leibler, ahora se hace con respecto a los puntos $\mathbf{y}_i^{(m)}$ y los pesos $\pi_i^{(m)}$ [^1]. Cabe resaltar que el modelo no es un modelo de mezclas con respecto a los mapas, pues en ese caso se usaría un peso por mapa para determinar su importancia; es más bien una mezcla con respecto a las similaridades entre objectos directamente. 

Este procedimiento permite representar relaciones no transitivas. Por ejemplo, en el caso de "lengua", "taco" y "española", supongamos que hay dos mapas, en ambos las tres palabras están cerca, y los pesos de importancia son

|               | $x=\mathrm{lengua}$ | $x=\mathrm{tacos}$ | $x=\mathrm{espa\tilde nola}$ |
| ------------- | ------------------- | ------------------ | ---------------------------- |
| $\pi_x^{(1)}$ | $1/3$               | $2/3$              | $0$                          |
| $\pi_x^{(2)}$ | $1/3$               | $0$                | $2/3$                        |

En este caso, la similaridad entre lengua y taco es más o menos (porque estamos
suponiendo que los $\mathbf{y}$ de las tres palabras quedan cerca) $1/3 \times
2/3 = 2/9$, al igual que la similaridad entre lengua y española. Sin embargo, la similaridad entre taco y española es cero.

Una ventaja más es que podemos representar de mejor manera la centralidad. Recordemos que para tener $k$ puntos equidistantes en $\mathbb{R}^p$, necesitamos $p\geq k-1$, por lo que t-SNE no puede representar las situaciones en las que más de tres (en el caso bidimensional) puntos tienen como más cercano a un mismo punto central. La extensión con mapas múltiples lo resuelve de la misma manera, asignando importancias cero en algunos mapas para conseguir que en el conjunto de todos los mapas se represente la centralidad. 

En la práctica surgen complicaciones como elegir $M$, el número de mapas. De manera similar a la elección de $k$ en kNN, la elección se puede hacer a través de manera gráfica. Para algún número predeterminado $M$, graficamos la razón de preservación de vecindades usando $m=1, \cdots, M$ mapas, y el comportamiento asintótico de la gráfica (pues eventualmente se capturó ya toda la estructura y no se necesitan más mapas) da un buen punto de corte. 

###### Definición 

La *razón de preservación de vecindades* es 
$$
\rho(m,k)=\frac{1}{n}\sum_{i=1}^n\frac{1}{\#\mathcal{N}_k(\mathbf{y}_i)}\sum_{\mathbf{y}_j\in\mathcal{N}_k(\mathbf{y}_i)}[\mathbf{x}_j\in\mathcal{N}_k(\mathbf{x}_i)]
$$


donde $\mathcal{N}_k(\mathbf{w})$ es el conjunto de los $k$ vecinos más cercanos a $\mathbf{w}$, y $[P]$ vale 1 si $P$ 					es cierto ó 0 si es falso. Es decir, para cada punto  $i$ se calcula la proporción dentro de sus $k$ vecinos más cercanos en $\mathbb{R}^q$ son también vecinos más cercanos en $\mathbb{R}^p$.  

Elegir  $M$ con esta medida corresponde a lo que un usuario haría normalmente: revisar todos los mapas en busca de relaciones significativas entre palabras, y quedarse sólo con las que le aportan algo. 

Al evaluar t-SNE con mapas múltiples hay que tener en mente la asimetría de la divergencia de Kullback-Leibler presentada en el apéndice A. La optimización no penaliza casos en los que puntos disimlares (con $p_{ij}$ pequeño) quedan juntos en el mapa (tienen $q_{ij}$ grande). Esta es una de las diferencias fundamentales entre t-SNE con mapas múltiples y los modelos de tópicos, pues que dos palabras tengan pesos de importancia similares en un mapa no significa que estén relacionadas. 

Sin embargo, ventajas sobre los modelos de tópicos son que t-SNE con mapas múltiples permite estudiar estructuras y relaciones sutiles entre las palabras que un modelo de tópicos no encontraría y puede entrenarse sólo con una matriz de disimilaridades. 

[^1]: En realidad se entrenan pesos $w_i^{(m)}$ sin restricciones para usar descenso en gradiente, y después basta usar $\pi_i^{(m)} \propto e^{-w_i^{(m)}}$.

#### t-SNE paramétrico

Otro problema de t-SNE es que no se extiende a nuevas observaciones. Supongamos que se corrió t-SNE sobre un conjunto de datos $X \in \mathcal{M}_{n\times p}(\mathbb{R})​$ y que recibimos una nueva observación $\mathbf{x}_{n+1}​$. ¿Cuáles deberían ser sus coordenadas en el nuevo espacio? t-SNE tradicional no da una manera de asignarlo porque es un método no-paramétrico; no hay manera de relacionar una nueva observación porque no estuvo en el proceso inicial. Los métodos paramétricos buscan dar una función explícita $f_w : \mathbb{R}^p \to \mathbb{R}^q​$ en términos del parámetro $w​$ para mapear nuevos puntos.

Laures van der Maaten propone en [4] una parametrización de t-SNE usando una red neuronal profunda entrenada en partes: primero se preentrena una pila de máquinas de Boltzmann restringidas (modelos gráficos bipartitos completos) como autoencoder (es decir, se entrena una subred para comprimir los datos a una dimensión menor y después reconstruirlos) y después se entrena con backpropagation una red que utiliza la salida del autoencoder como entrada.

Un método kernelizado fue propuesto por Gisbrecht, Shulz y Hammer en [5]. La función parametrizada toma la forma
$$
f_w(\mathbf{x}) =\sum_j\mathbf{\alpha}_j\frac{k(\mathbf{x}, \mathbf{x}_j)}{\sum_lk(\mathbf{x}, \mathbf{x}_l)}
$$
donde $k(\mathbf{x}, \mathbf{x}_j)=\exp\left(\frac{\|\mathbf{x}-\mathbf{x}_j\|^2}{2\sigma_j^2}\right)$ es el kernel normal y $\alpha_j\in\mathbb{R}^q$ son parámetros que dependen de los puntos en el espacio $q$ dimensional. Suponiendo que se corrió t-SNE sobre una muestra original y contamos con el $$\mathbf{y}\in\mathbb{R}^p$$ correspondiente a cada $\mathbf{x}\in\mathbb{R}^q$, basta encontrar los $\mathbf{\alpha}_j$ para tener el mapeo explícitamente.

###### Teorema

Sea $K\in\mathcal{M}_n(\mathbb{R})$ la matriz con entradas $(k)_{ij}= k(\mathbf{x}_i, \mathbf{x}_j)/\sum_lk(\mathbf{x}_i, \mathbf{x}_l)$ y $Y\in\mathcal{M}_{n\times q}(\mathbb{R})$ la matriz con las coordenadas de cada punto en el espacio dimensional por filas. La matriz
$$
A = K^\dagger Y
$$
tiene en la fila $j$ al parámetro $\mathbf{\alpha}_j \  _\square$.

###### Definición 

El parámetro $\sigma_j$ del kernel, llamado *ancho de banda* del kernel. 

Este resultado indica la manera de extender t-SNE a nuevas observaciones: lo corremos en la muestra inicial, calculamos $A$ y asignamos cada nueva observación usando $f_w$. 

###### Teorema

Si $\mathbf{x}_j$ es uno de los puntos en la muestra original, 
$$
\lim_{\sigma_j\rightarrow 0}\mathbf{\alpha}_j=\mathbf{y}_j
$$
por lo que $f_w(\mathbf{x}_j)=\mathbf{y}_j \ _\square$. 

Los autores recomiendan elegir el ancho de banda $\sigma_j$ como un múltiplo de la distancia de la observación $\mathbf{x}_j$ a su vecino más cercano.



