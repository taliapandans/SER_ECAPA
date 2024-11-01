import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np

## replaced all brackets [[ and ]] to " in the result data
## before running the code

df = pd.read_csv('./V21_model_4_0058_AAM.csv')
true_label = df['true_label']
pred_label = df['pred_label']
file_name = df['file_name']

for i, row in df.iterrows():
    embedding = np.fromstring(df.at[i,'embeddings'], dtype=float, sep=',')
    df.at[i,'embeddings'] = embedding

embeddings_df = df['embeddings']
# [np.darray, obj2]
embeddings = np.array([y for y in embeddings_df])

tsne = manifold.TSNE(n_components=2, random_state=42)
mnist_tr = tsne.fit_transform(embeddings)

# create dataframe
## visualize true or predicted label
# cps_df = pd.DataFrame(columns=['Dimension 1', 'Dimension 2', 'target'],
#                        data=np.column_stack((mnist_tr, 
#                                             true_label)))
cps_df = pd.DataFrame(columns=['Dimension 1', 'Dimension 2', 'target'],
                       data=np.column_stack((mnist_tr, 
                                            pred_label)))

# cast targets column to int
cps_df.loc[:, 'target'] = cps_df.target.astype(int)

emo_map = {0:'Neutral',
               1: 'Happy',
               2: 'Sad',
               3: 'Angry'}
emo_order = ['Neutral', 'Happy', 'Sad', 'Angry']

# map targets to actual clothes for plotting
cps_df.loc[:, 'target'] = cps_df.target.map(emo_map)

grid = sns.FacetGrid(cps_df, hue="target", hue_order = emo_order, height=5, aspect=1.5)
grid.map(plt.scatter, 'Dimension 1', 'Dimension 2')


plt.legend(loc='upper right')
plt.title('t-SNE Result: V21 Model 4_0058 Embedding 1 AAM (pred labels)', y=1.0, pad=-3)

plt.show()
