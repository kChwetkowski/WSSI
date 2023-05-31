import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Neuron:
    def __init__(self, n_inputs, bias = 0., weights = None):
        self.b = bias
        if weights: self.ws = np.array(weights)
        else: self.ws = np.random.rand(n_inputs)

    def _f(self, x): #activation function (here: leaky_relu)
        return max(x*.1, x)

    def __call__(self, xs): #calculate the neuron's output: multiply the inputs with the weights and sum the values together, add the bias value,
                            # then transform the value via an activation function
        return self._f(xs @ self.ws + self.b)


class NeuralNetwork:
    def __init__(self):
        self.input_layer = [Neuron(1) for _ in range(3)]
        self.hidden_layer1 = [Neuron(3) for _ in range(4)]
        self.hidden_layer2 = [Neuron(4) for _ in range(4)]
        self.output_neuron = Neuron(4)

    def forward(self, input_data):
        hidden1_outputs = [neuron(input_data) for neuron in self.hidden_layer1]
        hidden2_outputs = [neuron(hidden1_outputs) for neuron in self.hidden_layer2]
        output = self.output_neuron(hidden2_outputs)
        return output

def visualize_network():
    G = nx.DiGraph()

    G.add_nodes_from(['Input 1', 'Input 2', 'Input 3'])
    G.add_nodes_from(['Hidden 1-1', 'Hidden 1-2', 'Hidden 1-3', 'Hidden 1-4'])
    G.add_nodes_from(['Hidden 2-1', 'Hidden 2-2', 'Hidden 2-3', 'Hidden 2-4'])
    G.add_node('Output')

    G.add_edges_from([('Input 1', 'Hidden 1-1'), ('Input 1', 'Hidden 1-2'), ('Input 1', 'Hidden 1-3'), ('Input 1', 'Hidden 1-4')])
    G.add_edges_from([('Input 2', 'Hidden 1-1'), ('Input 2', 'Hidden 1-2'), ('Input 2', 'Hidden 1-3'), ('Input 2', 'Hidden 1-4')])
    G.add_edges_from([('Input 3', 'Hidden 1-1'), ('Input 3', 'Hidden 1-2'), ('Input 3', 'Hidden 1-3'), ('Input 3', 'Hidden 1-4')])
    G.add_edges_from([('Hidden 1-1', 'Hidden 2-1'), ('Hidden 1-1', 'Hidden 2-2'), ('Hidden 1-1', 'Hidden 2-3'), ('Hidden 1-1', 'Hidden 2-4')])
    G.add_edges_from([('Hidden 1-2', 'Hidden 2-1'), ('Hidden 1-2', 'Hidden 2-2'), ('Hidden 1-2', 'Hidden 2-3'), ('Hidden 1-2', 'Hidden 2-4')])
    G.add_edges_from([('Hidden 1-3', 'Hidden 2-1'), ('Hidden 1-3', 'Hidden 2-2'), ('Hidden 1-3', 'Hidden 2-3'), ('Hidden 1-3', 'Hidden 2-4')])
    G.add_edges_from([('Hidden 1-4', 'Hidden 2-1'), ('Hidden 1-4', 'Hidden 2-2'), ('Hidden 1-4', 'Hidden 2-3'), ('Hidden 1-4', 'Hidden 2-4')])
    G.add_edges_from([('Hidden 2-1', 'Output'), ('Hidden 2-2', 'Output'), ('Hidden 2-3', 'Output'), ('Hidden 2-4', 'Output')])

    pos = {'Input 1': (0, 2.5), 'Input 2': (0, 1.75), 'Input 3': (0, 1),
           'Hidden 1-1': (1.25, 3), 'Hidden 1-2': (1.25, 2.25), 'Hidden 1-3': (1.25, 1.5), 'Hidden 1-4': (1.25, 0.75),
           'Hidden 2-1': (2.5, 3), 'Hidden 2-2': (2.5, 2.25), 'Hidden 2-3': (2.5, 1.5), 'Hidden 2-4': (2.5, 0.75),
           'Output': (3.75, 1.75)}

    # kwestie wizualne
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        node_color='white',
        node_size=1500,
        font_weight='bold',
        font_color='black',
        edge_color='gray',
        node_shape='o',
        linewidths=1,
        edgecolors='black'
    )

    layers = [['Input 1', 'Input 2', 'Input 3'], ['Hidden 1-1', 'Hidden 1-2', 'Hidden 1-3', 'Hidden 1-4'],
              ['Hidden 2-1', 'Hidden 2-2', 'Hidden 2-3', 'Hidden 2-4'], ['Output']]
    layer_labels = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
    for i, layer in enumerate(layers):
        x = pos[layer[0]][0] - 0.25
        y = 0.25
        plt.text(x, y, layer_labels[i], fontsize=10)

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 3.5)
    plt.axis('off')

    plt.show()

network = NeuralNetwork()

visualize_network()
