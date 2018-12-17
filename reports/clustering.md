## Clustering y convergencia

t-SNE es una de las herramientas más utilizadas para realizar reducción de dimensión y visualización de datos en altas dimensiones, en general las visualizaciones producidas muestran la formación de clusters naturales en los datos, sin embargo, tanto los autores del método como la comunidad estadística en general no utilizan algoritmos de clusterización sobre resultados obtenidos por t-SNE puesto que no se tienen resultados teóricos sobre la estructura de el mapeo producido.

A continuación presentamos un estudio realizado por Linderman y Steinerberger en el 2017 que presenta condiciones ideales sobre los datos asi como requisitos sobre la elección óptima parámetros para asegurar la convergencia y la formación de clusters.

###### Recordatorio

Comencemos por expresar el gradiente de la función costo en de la misma forma que en Barnes-Hut-SNE, en función de fuerzas atractivas y repulsivas.

$$
\nabla_{y_i} C = 4 (F_{a} - F_{r})
$$
Asimismo, introduzcamos el coeficiente de exageración $\beta > 1$ correspondiente a el método de descenso por gradiente.
$$
\frac{1}{4}\nabla_{y_i} C = \beta F_{a} - F_{r}
$$
Finalmente, introducimos un paso de tamaño $h > 0 $
$$
\frac{h}{4}\nabla_{y_i} C = \beta F_{a} - F_{r}
$$

###### Suposiciones

1. *X está cluterizado*: existe un número natural k (el número de clusters) y un mapeo que asigna a cada punto a uno de los clusters $\pi : \{1,...n\} \to \{1,...,k\}$  y que cumple

$$
p_{ij} \geq \frac{1}{10n\vert \pi^{-1}(\pi(i))\vert} \quad \text{si} \ \pi(x_i) = \pi(x_j)
$$
​    Observemos que $\vert \pi^{-1}(\pi(i))\vert$ no es más que el tamaño del cluster en que se encuentra $x_i$.

2. *Elección de parámetros*: $\beta$ y $h$ son elegidos de forma que para algún $i \in \{1,...,n\}$ 

$$
\frac{1}{100} \leq \alpha h, \quad \quad \sum_{\substack{j \not = i \\\pi(j)= \pi(i)}}p_{ij} \leq \frac{9}{10}
$$
3. *Inicialización:* la inicialización cumple $\mathcal{Y}^0 \subset [-.01, .01]^2$. Esta suposición puede ser modificada pero en general, es mejor la inicialización en valores pequeños.

###### Teorema

Sea $\mathcal{C}_i$ el i-ésimo cluster, es decir $\mathcal{C}_i := \{y_i | 1 \leq j \leq n \ y \ \pi(j) = \pi(i) \}$. El diámetro del cluster $\mathcal{C}_i$ decae exponencialmente (a una tasa universal) hasta satisfacer
$$
diam(\mathcal{C}_i) \leq c \ h\left( \beta \sum_{\substack{j \not = i \\\pi(j)\not= \pi(i)}}p_{ij} \right)
$$
Para alguna constante $c > 0 $

> *Demostración*
>
> Ver paper

###### Algunos resultados

En general, $\beta \sim \frac{hn}{10}$  es una configuración deseable de parámetros, en particular la combinación canónica elegida por los autores es
$$
\beta \sim \frac{n}{10} \quad \quad h \sim 1
$$
Elecciones con dicha configuración conducen a una tasa de convergencia exponencial con tasa $\kappa$
$$
\kappa \sim 1 - \frac{\beta h}{n}
$$
Notemos que si $\beta h \geq n $ se rompe la convergencia del algoritmo. 

La elección de parámetros por *regla de dedo* para t-SNE es $\beta \sim 12$ y $h \sim 200$, notemos que para $n\leq 24000$ cumple con la configuración propuesta, sin embargo, si el número de observaciones es mayor se viola la cota inferior correspondiente desacelerando la convergencia.

La mejor tasa de convergencia se obtiene con la siguiente selección de parámetros
$$
\beta h = \frac{9}{10} \left( \sum_{\substack{j \not = i \\\pi(j)= \pi(i)}}p_{ij} \right)^{-1}
$$
Asimismo, la mejor elección de parámetros que garantiza la mejor convergencia posible de los clusters es
$$
\beta h = \frac{9}{10} \left( \max_{1\leq i \leq n}\sum_{\substack{j \not = i \\\pi(j)= \pi(i)}}p_{ij} \right)^{-1}
$$
