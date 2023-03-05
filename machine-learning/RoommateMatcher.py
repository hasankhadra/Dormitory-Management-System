from keras.layers import Input, Dense, Lambda
from keras.models import Model
import keras.backend as K

class SiameseNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self._build_model()

    def _build_model(self):
        """
        Builds a Siamese network with two identical subnetworks
        """
        # Define the input layers
        input_left = Input(shape=(self.input_dim,))
        input_right = Input(shape=(self.input_dim,))

        # Define the shared subnetwork
        shared_network = Dense(64, activation='relu')

        # Connect the input layers to the shared network
        output_left = shared_network(input_left)
        output_right = shared_network(input_right)

        # Compute the distance between the outputs of the shared network
        distance = Lambda(SiameseNetwork.euclidean_distance)([output_left, output_right])

        # Define the final output layer
        output = Dense(1, activation='sigmoid')(distance)

        # Create the Siamese network
        self.model = Model(inputs=[input_left, input_right], outputs=output)

    @staticmethod
    def euclidean_distance(inputs):
        """
        Computes the Euclidean distance between two inputs
        """
        x, y = inputs
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()

    def save(self, filename):
        self.model.save(filename)