import numpy as np
import pandas as pd
import os
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import scipy

def create_adjacency_matrix(X, columns, th=10):
    df_data_train = pd.DataFrame(X, columns=columns)
    df_G = pd.DataFrame(columns=['from', 'to', 'weight'])

    for i in df_data_train.columns[:-1]:
        max_val = df_data_train[i].max()
        df_aux_i = df_data_train[df_data_train[i] > (max_val - th)]
        df_aux_i = df_aux_i.drop(i, axis=1)

        dfs = []
        for k, v in df_aux_i.mean().items():
            dfs.append(pd.DataFrame({'from': i, 'to': k, 'weight': v}, index=[0]))

        df_G = pd.concat(dfs, ignore_index=True)

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
BUILDING_UNDER_ANALYSIS_ID = 1
df = df[df.BUILDINGID == BUILDING_UNDER_ANALYSIS_ID]

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
one_hot_encoder = OneHotEncoder(sparse_output=False)
Y = one_hot_encoder.fit_transform(df_y.values.reshape(-1, 1))
num_classes = Y.shape[1]

print(f"{Y.shape=}")
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
    def __init__(self, data, **kwargs):
        self.data = data
        super().__init__(**kwargs)

    def read(self):
        x = self.data['X'].astype('float32')
        a = self.data['A']
        y = self.data['Y']
        
        graphs = []
        for i in range(len(x)):
            graphs.append(Graph(x=x[i], a=sp.csr_matrix(a), y=y[i]))
        return graphs

dataset = UJIIndoorLocDataset(np.load('data/UJIIndoorLoc/preprocessed_data.npz'))

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.data.loaders import DisjointLoader
from spektral.layers import GATConv, GCSConv, GlobalSumPool
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from tensorflow import keras
keras.config.disable_traceback_filtering()

# Parameters
channels = 8
n_attn_heads = 8
dropout = 0.6
l2_reg = 2.5e-4
learning_rate = 5e-3
epochs = 500
es_patience = 10  # Patience for early stopping
batch_size = 1

# Train/test split
np.random.shuffle(dataset)
split = int(0.8 * len(dataset))
dataset_tr, dataset_va = dataset[:split], dataset[split:]

# Data loaders
loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size)

# Model definition
class IndoorLocGAT(Model):
    def __init__(self, n_labels):
        super().__init__()
        self.conv1 = GCSConv(256, activation="relu")
        self.conv2 = GCSConv(128, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.conv4 = GATConv(n_labels)
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_labels, activation="sigmoid")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        output = self.global_pool([x, tf.cast(i, tf.int32)])
        output = self.dense(output)
        return output

model = IndoorLocGAT(dataset_tr.n_labels)

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)
# Print a summary of the model architecture
model.summary()

# eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
# print("Before training: .\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

# model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, verbose=1, validation_data=loader_va.load(), validation_steps=loader_va.steps_per_epoch, epochs = epochs)

# eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
# print("Done training.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy(reduction="sum")

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
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

################################################################################
# Evaluate model
################################################################################
# Load the dataset
df_test = pd.read_csv('FinalProject/ValidationData.csv')

# Use only one building data
df_test = df_test[df_test.BUILDINGID == BUILDING_UNDER_ANALYSIS_ID]

# How many floors does this build have.
# print(f"Building {BUILDING_UNDER_ANALYSIS_ID} floors {df.groupby('FLOOR').count()} Count")

# Replace 100 with -105 for RSSI values
df_test.iloc[:, :520] = df_test.iloc[:, :520].replace(100, -105)

# Normalize RSSI values
df_test.iloc[:, :520] += 105

# Extract features and labels
df_test_X = df_test.iloc[:,:520]
df_test_y = df_test['FLOOR']

df_test_X = df_test_X[ap_buil_2]
df_test_X.describe()
X_test = df_test_X.values

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)
Y_test = one_hot_encoder.fit_transform(df_test_y.values.reshape(-1, 1))
num_classes = Y_test.shape[1]

print(f"{Y_test.shape=}")
print(f"{X_test.shape=}")

# Create adjacency matrix
G_test, A_test = create_adjacency_matrix(X_test, columns)

# Save preprocessed data
np.savez('data/UJIIndoorLoc/preprocessed_test_data.npz', X=X_test, A=A_test, Y=Y_test)


dataset_te = UJIIndoorLocDataset(np.load('data/UJIIndoorLoc/preprocessed_test_data.npz'))

loader_te = DisjointLoader(dataset_te, batch_size=batch_size)

model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))