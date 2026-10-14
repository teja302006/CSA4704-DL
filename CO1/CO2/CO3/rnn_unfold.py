# Experiment 14: Unfolded RNN Architecture

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(10)
tf.random.set_seed(10)

X = np.random.rand(1, 5, 2).astype("float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 2)),
    tf.keras.layers.SimpleRNN(3, return_sequences=True),
    tf.keras.layers.Dense(1)
])

model.predict(X, verbose=0)

rnn = model.layers[0]
dense = model.layers[1]

Wx, Wh, bh = rnn.get_weights()
Wy, by = dense.get_weights()

hidden = np.zeros((1, 3))
states = []

for t in range(5):
    hidden = np.tanh(X[:, t, :] @ Wx + hidden @ Wh + bh)
    states.append(hidden.ravel())

states = np.array(states)
output = states @ Wy + by

print("Manual Outputs:\n", output.ravel())
print("Keras Outputs:\n", model.predict(X, verbose=0).ravel())

G = nx.DiGraph()

for t in range(1, 6):
    G.add_node(f"x{t}", color="lightblue")
    G.add_node(f"h{t}", color="lightgreen")
    G.add_node(f"y{t}", color="salmon")
    G.add_edge(f"x{t}", f"h{t}", label="Wxh")
    G.add_edge(f"h{t}", f"y{t}", label="Why")

    if t > 1:
        G.add_edge(f"h{t-1}", f"h{t}", label="Whh")

pos = {}
for t in range(1, 6):
    pos[f"x{t}"] = (t, 2)
    pos[f"h{t}"] = (t, 1)
    pos[f"y{t}"] = (t, 0)

colors = [G.nodes[n]["color"] for n in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1400)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
plt.title("RNN Unfolded Across Time")
plt.show()

plt.plot(range(1, 6), states)
plt.xlabel("Time Step")
plt.ylabel("Hidden State")
plt.title("Hidden State Values")
plt.show()
