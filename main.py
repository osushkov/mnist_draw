import math
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
import idx

IMAGE_DIM = 28
INPUTS_FILENAME = "inputs.dat"
OUTPUTS_FILENAME = "outputs.dat"

class DrawNet(snt.AbstractModule):
    def __init__(self, layer_sizes, layer_activations, output_activation):
        super(DrawNet, self).__init__(name='draw_network')

        self._network = snt.nets.MLP(layer_sizes, activation=layer_activations)
        self._value = snt.Linear(output_size=1)
        self._output_activation = output_activation

    def _build(self, pixel_pos, embedding):
        embedding_flattened = snt.BatchFlatten()(embedding)
        pixel_pos_flattened = snt.BatchFlatten()(pixel_pos)

        net_in = tf.concat([embedding_flattened, pixel_pos_flattened], 1)
        net_out = self._network(net_in)
        return self._output_activation(self._value(net_out))


def drawImage(img):
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.show()


def loadTrainingData(inputs_filename, outputs_filename):
    inputs_file = open(inputs_filename, "r")
    outputs_file = open(outputs_filename, "r")
    return np.load(inputs_file), np.load(outputs_file)


def saveTrainingData(inputs, inputs_filename, outputs, outputs_filename):
    inputs_file = open(inputs_filename, "w")
    outputs_file = open(outputs_filename, "w")

    np.save(inputs_file, inputs)
    np.save(outputs_file, outputs)


def genTrainingData(img_path, label_path):
    img_data = idx.readImages(img_path).reshape((-1, IMAGE_DIM, IMAGE_DIM))
    label_data = idx.readLabels(label_path)

    sample_input = np.zeros((img_data.shape[0] * IMAGE_DIM * IMAGE_DIM, 12), dtype=np.float32)
    sample_output = np.zeros((img_data.shape[0] * IMAGE_DIM * IMAGE_DIM, 1), dtype=np.float32)

    index = 0
    for i in range(img_data.shape[0]):
        print(str(i))
        for yi in range(IMAGE_DIM):
            for xi in range(IMAGE_DIM):
                x = float(xi) / float(IMAGE_DIM)
                y = float(yi) / float(IMAGE_DIM)
                pix = img_data[i, yi, xi]

                sample_input[index,:] = np.concatenate([np.array([x, y]), label_data[i]])
                sample_output[index] = pix
                index += 1
        # drawImage(img_data[i])

    permutation = np.random.permutation(sample_input.shape[0])
    return sample_input[permutation], sample_output[permutation]


try:
    training_data = loadTrainingData(INPUTS_FILENAME, OUTPUTS_FILENAME)
except IOError:
    training_data = genTrainingData("data/train_images.idx3", "data/train_labels.idx1")
    saveTrainingData(training_data[0], INPUTS_FILENAME, training_data[1], OUTPUTS_FILENAME)

print("training data shape: {} {}".format(training_data[0].shape, training_data[1].shape))