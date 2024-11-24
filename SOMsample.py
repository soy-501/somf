!pip install somoclu
!pip install japanize-matplotlib
import japanize_matplotlib

import numpy as np
import pandas as pd

from somoclu import Somoclu
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from google.colab import drive
drive.mount('/content/drive')

# データを読み込む
df_pokemon = pd.read_csv('/content/drive/MyDrive/MyWorks/data/pokemon_data.csv', encoding='shift_jis')
df_pokemon

X = df_pokemon.drop(['id', '名前', '画像URL'], axis=1)
X = pd.get_dummies(X)
y = df_pokemon['名前']
   
# SOMに入れる前にPCA(主成分分析)して計算コスト削減を測る 
pca = PCA(n_components=5) 
X = pca.fit_transform(X)

# SOMの定義
n_rows = 16
n_cols = 24
som = Somoclu(n_rows=n_rows, n_columns=n_cols,
              initialization="pca", verbose=2, compactsupport=False)

# 学習

%%time
som.train(data=X, epochs=1000)
