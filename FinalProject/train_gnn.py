"""
This example implements the model from the paper

    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You, Rex Ying, Jure Leskovec

using the PROTEINS dataset.

The configuration at the top of the file is the best one identified in the
paper, and should work well for many different datasets without changes.

Note: the results reported in the paper are averaged over 3 random repetitions
with an 80/20 split.
"""
import numpy as np
import pandas as pd
import os
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import scipy

def create_adjacency_matrix(X, columns, th=5):
    df_data_train = pd.DataFrame(X, columns=columns)
    edges = []

    for i in df_data_train.columns[:-1]:
        max_val = df_data_train[i].max()
        df_aux_i = df_data_train[df_data_train[i] > (max_val - th)].drop(i, axis=1)
        
        for k, v in df_aux_i.mean().items():
            edges.append((i, k, v))
    
    df_G = pd.DataFrame(edges, columns=['from', 'to', 'weight'])
    G = nx.from_pandas_edgelist(df_G, source='from', target='to', edge_attr='weight')
    W = nx.to_numpy_array(G)
    np.fill_diagonal(W, 0)
    
    # Normalize the adjacency matrix
    (w, v) = scipy.sparse.linalg.eigs(W, k=1, which='LM')
    W = W / np.abs(w[0])

    return G, W


# Load the dataset
df = pd.read_csv('FinalProject/TrainingData.csv')

# Use only one building data
BUILDING_UNDER_ANALYSIS_ID = 0
df = df[df.BUILDINGID == BUILDING_UNDER_ANALYSIS_ID]

# How many floors does this build have.
# print(f"Building {BUILDING_UNDER_ANALYSIS_ID} floors {df.groupby('FLOOR').count()} Count")

# Replace 100 with -105 for RSSI values
df.iloc[:, :520] = df.iloc[:, :520].replace(100, -105)

# Normalize RSSI values
df.iloc[:, :520] += 105

# Extract features and labels
df_X = df.iloc[:,:520]
df_y = df['FLOOR']

# Ignore WAP where RSSI std deviation is lower than 5
print(df_X.describe())
ap = (df_X.describe().iloc[2]>5).index
values = (df_X.describe().iloc[2]>5).values
ap_buil_2 = [ap[i] for i in range(len(values)) if values[i]==True]
df_X = df_X[ap_buil_2]
print(df_X.shape)
df_X.describe()


df_X.hist(figsize=[30,30])
plt.show()
df_y.hist(figsize=[10,10])
plt.show()

X = df_X.values

from sklearn.preprocessing import OneHotEncoder

# Assuming df_y contains the class labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
Y = one_hot_encoder.fit_transform(df_y.values.reshape(-1, 1))
num_classes = Y.shape[1]  # Number of classes should be 5

print(f"{Y.shape=}")  # Should be (number of samples, 5)
print(f"{X.shape=}")

# Create adjacency matrix
columns = df_X.columns
G, A = create_adjacency_matrix(X, columns)

# Save preprocessed data
os.makedirs('data/UJIIndoorLoc', exist_ok=True)
np.savez('data/UJIIndoorLoc/preprocessed_data.npz', X=X, A=A, Y=Y)

from spektral.data import Dataset, Graph
import scipy.sparse as sp

class UJIIndoorLocDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        data = np.load('data/UJIIndoorLoc/preprocessed_data.npz', allow_pickle=True)
        x = data['X']
        a = data['A']
        y = data['Y']
        
        graphs = []
        for i in range(len(x)):
            graphs.append(Graph(x=x[i], a=sp.csr_matrix(a), y=y[i]))
        return graphs

dataset = UJIIndoorLocDataset()

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.models import GeneralGNN

################################################################################
# Config
################################################################################
batch_size = 1
learning_rate = 0.01
epochs = 20

################################################################################
# Load data
################################################################################
data = UJIIndoorLocDataset()

# Train/test split
np.random.shuffle(data)
split = int(0.8 * len(data))
data_tr, data_te = data[:split], data[split:]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

################################################################################
# Build model
################################################################################
model = GeneralGNN(num_classes, activation="softmax")
optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()

################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc

def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

epoch = step = 0
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        results_te = evaluate(loader_te)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                epoch, *np.mean(results, 0), *results_te
            )
        )
        results = []

################################################################################
# Evaluate model
################################################################################
results_te = evaluate(loader_te)
print("Final results - Loss: {:.3f} - Acc: {:.3f}".format(*results_te))