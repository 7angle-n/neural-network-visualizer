import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
import networkx as nx
from matplotlib.patches import Patch

st.set_page_config(page_title="TF Neural Network Playground", layout="wide")

#---------------------------------------
#Model
#---------------------------------------
def make_model(layer_sizes, activations, lr=0.01):
    model = tf.keras.Sequential()
    for i in range(len(layer_sizes)-1):
        model.add(tf.keras.layers.Dense(layer_sizes[i+1],
                                        activation=activations[i],
                                        input_shape=(layer_sizes[i],) if i==0 else []))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

#---------------------------------------
#Dataset
#---------------------------------------
def make_data(dataset="moons", noise=0.2, test_size=0.3, seed=42):
    rng = np.random.RandomState(seed)
    if dataset == "moons":
        X, y = make_moons(n_samples=1000, noise=noise, random_state=rng)
    elif dataset == "circles":
        X, y = make_circles(n_samples=1000, noise=noise, factor=0.5, random_state=rng)
    else:  # XOR
        X = rng.rand(1000, 2)
        y = ((X[:,0] > 0.5) ^ (X[:,1] > 0.5)).astype(int)
        X += rng.normal(0, noise, X.shape)
    return train_test_split(X, y, test_size=test_size, random_state=rng)

#---------------------------------------
#Decision Boundary
#---------------------------------------
def plot_boundary(model, X, y, title='Decision Boundary', resolution=200):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid, verbose=0)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.contourf(xx, yy, Z, cmap="RdBu", alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=10, edgecolor="k")
    ax.set_title(title)
    return fig

#---------------------------------------
#Visualize Network
#---------------------------------------
def visual_nn(layer_sizes, activations):
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    G = nx.DiGraph()
    pos, labels, colors = {}, {}, []

    act_color = {
        "input": "#a9a9a9",     # gray
        "relu": "#90ee90",      # light green
        "sigmoid": "#ffa500",   # orange
        "tanh": "#00bfff",      # blue
        "linear": "#ffff99",    # yellow
        "softmax": "#dda0dd",   # purple
        "none": "#ffffff"       # white
    }

    x_spacing, y_spacing = 2.0, 1.5
    num_layers = len(layer_sizes)

    node_list = [] 
    for layer_idx, layer_size in enumerate(layer_sizes):
        for neuron_idx in range(layer_size):
            node_id = f"L{layer_idx}N{neuron_idx}"
            x = layer_idx * x_spacing
            y = -neuron_idx * y_spacing + (layer_size - 1) * y_spacing / 2
            pos[node_id] = (x, y)
            node_list.append(node_id)
            G.add_node(node_id)

            if layer_idx == 0:
                act = "input"
                label = f"x{neuron_idx+1}"
            else:
                if (layer_idx - 1) < len(activations):
                    act = activations[layer_idx - 1].strip().lower()
                else:
                    act = "none"
                label = act

            labels[node_id] = label
            colors.append(act_color.get(act, "#ffffff"))

    # add edges
    for l in range(1, num_layers):
        for target_idx in range(layer_sizes[l]):
            target_id = f"L{l}N{target_idx}"
            for source_idx in range(layer_sizes[l-1]):
                source_id = f"L{l-1}N{source_idx}"
                G.add_edge(source_id, target_id)

    # Draw - ensure node order matches color order
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
            nodelist=node_list, node_color=colors, node_size=1200, font_size=10,
            edge_color="#888", arrows=True)

    ax.set_title("Neural Network Architecture", fontsize=14)
    ax.axis("off")

    # Only show legend entries for activations actually used
    used_acts = sorted(set([l for l in labels.values() if l in act_color]))
    legend_elements = [Patch(facecolor=act_color[a], label=a.capitalize()) for a in used_acts]
    ax.legend(handles=legend_elements, loc="upper right", title="Activations")

    return fig

#---------------------------------------
#Streamlit UI
#---------------------------------------
st.title("Neural Network Playground using Tensorflow")

with st.sidebar:
    dataset = st.selectbox("Select Dataset type: ", ["moons", "circles", "xor"])
    noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.01)
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    seed = st.number_input("Seed", 0, 9999, 42)

    hidden_layers = st.text_input("Hidden layer sizes (comma)", "8,8")
    hidden_sizes = [int(s) for s in hidden_layers.split(",")] if hidden_layers.strip() else []
    layers = [2] + hidden_sizes + [2]
    act_choices = ["relu", "tanh", "sigmoid", "linear"]
    activations = []
    for i in range(len(layers)-1):
        act = st.selectbox(f"Activation L{i+1}", act_choices, index=0, key=f"act_{i}")
        activations.append(act)

    lr = st.number_input("Learning rate", 0.0001, 1.0, 0.01, format="%.4f")
    epochs = st.slider("Epochs", 1, 200, 50, 1)
    batch_size = st.slider("Batch size", 8, 256, 64, 8)

with st.expander("Show Network Graph"):
    fig = visual_nn(layers, activations)
    st.pyplot(fig)

#---------------------------------------
# Training
#---------------------------------------
X_train, X_test, y_train, y_test = make_data(dataset, noise, test_size, seed)
model = make_model(layers, activations, lr)
history = model.fit(X_train, y_train, epochs=epochs,
                    batch_size=batch_size, verbose=0,
                    validation_data=(X_test, y_test))

#---------------------------------------
#Layout
#---------------------------------------
col1, col2 = st.columns(2)
with col1:
    fig = plot_boundary(model, X_train, y_train)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Training metrics")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(history.history["loss"], label="Loss")
    ax2.plot(history.history["accuracy"], label="Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

st.caption(f"Layers: {layers} | Activations: {activations} | lr={lr} | epochs={epochs} | batch={batch_size}")
