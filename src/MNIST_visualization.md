---
jupyter:
  jupytext:
    formats: ipynb,md:markdown
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.4
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
    version: 3.7.0
---

```python
from sklearn.datasets import *
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sn
import scipy

import subprocess
import pickle
import os

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.palettes import d3
import bokeh.models as bmo
```

#### Downloads MNIST dataset

```python
def download_mnist(output_file = 'data/mnist.pkl'):
    
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    
    cmd = ['curl', url, '--output', output_file + '.gz']
    subprocess.call(cmd)

    cmd = ['gunzip', output_file]
    subprocess.call(cmd)
    
download_mnist()
```

#### Loads dataset 

```python
with open('data/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
```

#### Runs t-SNE

```python
n = 50000
tsne = TSNE(n_components=2,random_state=1000,perplexity=35)
x_tsne = tsne.fit_transform(train_set[0][:n])
```

#### Makes visualization

```python
output_file("../resources/img/t-SNE_MNIST2.html")

url = 'https://s3-us-west-2.amazonaws.com/mnist-imgs/mnist_{}.jpg'
image_paths = [url.format(i) for i in range(n)]
source = ColumnDataSource(
        data=dict(
            x = x_tsne[:,0],
            y = x_tsne[:,1],
            value = [str(i) for i in train_set[1][:n]],
            pics = image_paths[:n]
        )
       )

hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@pics" height="80" alt="@imgs" width="80"
                ></img>
            </div>
        """
    )

title = 't-SNE visualization of MNIST dataset'
p = figure(plot_width=1000, plot_height=1000,title=title)

palette = d3['Category10'][len(set(train_set[1][:n]))]
factors = [str(i) for i in range(10)]

color_map = bmo.CategoricalColorMapper(factors= factors, palette=palette)
color_mapper = {'field': 'value', 'transform': color_map}

p.scatter(x='x', y='y', source=source, color = color_mapper)
p.add_tools(hover)

show(p)
```
