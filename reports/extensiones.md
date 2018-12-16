## Extensiones

#### Multiple maps t-SNE

Las similaridades $  q_{ij}$que se usan en t-SNE tienen una limitante que puede pasar desapercibida porque usualmente es algo que queremos: las propiedades de la métrica involucrada. Supongamos por ejemplo que los datos a visualizar son texto, y medimos la similaridad usando asociación entre palabras. Podría ser el caso, por ejemplo, que "lengua" tenga una alta similaridad a "tacos", pero también a "española". En este caso, t-SNE va a colocar a "española" y "tacos" más cerca de lo que en realidad deberían estar. Este efecto es inevitable por la construcción de las $q_{ij}$, que utiliza la distancia euclidiana y obliga así, con la desigualdad del triángulo, a tener resultados transitivos.

Una extensión a t-SNE presentada por van der Maarten y Hinton en [3] construye $M$  *mapas*, realizaciones de t-SNE con todas las palabras que asigna a cada punto una importancia. Formalmente, la *importancia* del punto $\mathbf{x}_ i$ en el mapa $m$ es $\pi_i^{(m)}$ con las restricciones $\forall i \forall m\ \pi_i^{(m)} \geq 0$  y  $\sum_m\pi_i^{(m)}=1$. Las nuevas simiaridades en el espacio pequeño están dadas por
$$
q_{ij}\propto\sum_m\pi_i^{(m)}\pi_j^{(m)}\left(1+\left\|\mathbf{y}_i^{(m)}-\mathbf{y}_j^{(m)}\right\|^2\right)^{-1}
$$
con la constante de normalización apropiada para que sumen uno. Al optimizar la divergencia de Kullback-Leibler, ahora se hace con respecto a los puntos $\mathbf{y}_i^{(m)}$ y los pesos $\pi_i^{(m)}$ [^1]. Cabe resaltar que el modelo no es un modelo de mezclas con respecto a los mapas, pues en ese caso se usaría un peso por mapa para determinar su importancia; es más bien una mezcla con respecto a las similaridades entre objectos directamente. 

[^1]: En realidad se entrenan pesos $w_i^{(m)}$ sin restricciones para usar descenso en gradiente, y después basta usar $\pi_i^{(m)} \propto e^{-w_i^{(m)}}$.

#### t-SNE paramétrico

Otro problema de t-SNE es que no se extiende a nuevas observaciones. Supongamos que se corrió t-SNE sobre un conjunto de datos $X \in \mathcal{M}_{n\times p}(\mathbb{R})$ y que recibimos una nueva observación $\mathbf{x}_{n+1}$. ¿Cuáles deberían ser sus coordenadas en el nuevo espacio? t-SNE tradicional no da una manera de asignarlo porque es un método no-paramétrico; no hay manera de relacionar una nueva observación porque no estuvo en el proceso inicial.

Laures van der Maaten propone en [4] una parametrización de t-SNE usando una red neuronal profunda entrenada en partes: primero se preentrena una pila de máquinas de Boltzmann restringidas (modelos gráficos bipartitos completos) y después se entrena con backpropagation.

Un método menos efectivo pero mucho más sencillo fue propuesto por Gisbrecht, Shulz y Hammer en [5]. 

