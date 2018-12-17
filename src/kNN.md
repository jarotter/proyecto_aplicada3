---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.1
---

# La maldición de la dimensionalidad

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.stats import uniform
from sklearn.neighbors import NearestNeighbors
```

```python
n = 100
neigh = NearestNeighbors(n_neighbors=20)
distances = pd.DataFrame()
for p in (2, 8, 32, 128):
    x = np.array([uniform.rvs(size = n) for i in range(p)]).T
    neigh.fit(x)
    d = neigh.kneighbors(x)[0].flatten()
    d = d[d>0]
    sns.kdeplot(np.log(d), shade=True, label=p)
plt.title('Log de distancia a 20 vecinos más cercanos')
```

```python

```
