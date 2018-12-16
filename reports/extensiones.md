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

