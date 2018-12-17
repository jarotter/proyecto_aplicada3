## Apéndice B: Definiciones de geometría

###### Definición

Una *variedad p-dimensional* es un espacio Hausdorff paracompacto localmente homeomorfo a $ \mathbb{R} ^p$. A cada homeomorfismo se le llama *carta*, y al conjunto de ellas *atlas*. 

> Explicación:
>
> Una variedad es un espacio localmente plano. Por ejemplo, aunque la tierra sea una esfera no nos damos cuenta a menos que veamos a un horizonte sin estorbos, pues cerca de nosotros parece que vivimos en un mundo plano.



###### Definición

Un *mapa de transición* es la composición de una carta $\varphi_x$ con la inversa de otra carta $.\varphi_y^{-1}$

> Explicación:
>
> Los mapas de transición definen homeomorfismos entre abiertos de la variedad. La idea es pasar de la variedad al espacio plano y de ahí regresar a otro abierto en la variedad. 



###### Definición

Una variedad es *suave* si todos sus mapas de transición son de clase $C^\infty$. 



###### Definición

Sea $\mathcal{M}$ una variedad suave y $ x \in \mathcal{M}$. El *espacio tangente a* $\mathcal{M}$ *en* $ x$ , $T_x\mathcal{M}$ es el conjunto  de todos los vectores tangentes a $x$

> Explicación:
>
> Aunque es posible formalizar el concepto para que no dependa del espacio ambiente, podemos pensar que una variedad $q$-dimensional está embedida en un espacio con dimensión $p$ más grande e imaginar $T_x\mathcal{M}$ como el hiperplano afín tangente a $\mathcal{M}$ en $x$.



###### Definición

Una *variedad riemanniana* es una veriedad suave tal que

1. Para todo $x\in\mathcal{M}$ hay un producto interno $g_x$ en $T_x\mathcal{M}$.
2. Para todos campos vectoriales diferenciables $X, Y$ en $\mathcal{M}$, el mapa $x\mapsto g_x(X|x, Y|x)$ es suave.

> Explicación:
>
> En cada punto de la variedad hay un espacio con su producto interno, y si nos movemos a lo largo de la variedad, la transición de un producto interno a otro es suave.



###### Definición

El conjunto $\{ g_x : x \in \mathcal{M}\}$ es una *métrica riemanniana en* $\mathcal{M}$.

> Explicación:
>
> Para medir la distancia entre dos puntos $x$ y $x'$ inducida por los productos internos, necesitamos ir cambiando de producto interno en el camino. Esta métrica se va adaptando a la geometría del espacio conforme nos recorremos.

