import numpy as np
from RoommateMatcher import SiameseNetwork
# Define the number of input vectors and the number of dimensions of each vector
num_inputs = 100
input_dim = 10

# Generate random input vectors
input_vectors = np.random.rand(num_inputs, input_dim)

# Create pairs of input vectors with their similarity scores
num_pairs = int(num_inputs * (num_inputs - 1) / 2)  # Number of possible pairs
input_pairs = np.zeros((num_pairs, 2, input_dim))
labels = np.zeros((num_pairs,))
pair_index = 0
for i in range(num_inputs):
    for j in range(i+1, num_inputs):
        input_pairs[pair_index, 0] = input_vectors[i]
        input_pairs[pair_index, 1] = input_vectors[j]
        similarity_score = np.dot(input_vectors[i], input_vectors[j]) / (np.linalg.norm(input_vectors[i]) * np.linalg.norm(input_vectors[j]))
        labels[pair_index] = similarity_score
        pair_index += 1



# # Create the Siamese network
# siamese_network = create_siamese_network(input_dim=10)

# # Compile the model with a binary cross-entropy loss function and an Adam optimizer
# siamese_network.compile(loss='binary_crossentropy', optimizer='adam')

# # Train the model on pairs of input vectors and their corresponding similarity scores
# siamese_network.fit([input_pairs[:, 0], input_pairs[:, 1]], labels, epochs=10, batch_size=32)

siamese_network = SiameseNetwork(input_dim=10)

siamese_network.compile(optimizer='adam', loss='binary_crossentropy')

siamese_network.model.fit([input_pairs[:, 0], input_pairs[:, 1]], labels, epochs=10, batch_size=32)

# Generate two input vectors to compare
input_1 = np.random.rand(input_dim)
input_2 = np.random.rand(input_dim)

# Reshape the input vectors to have shape (1, input_dim) for compatibility with the model
input_1 = input_1.reshape(1, input_dim)
input_2 = input_2.reshape(1, input_dim)

# Compute the similarity score between the input vectors using the Siamese network
similarity_score = siamese_network.model.predict([input_1, input_2])[0, 0]

print("The similarity score between the input vectors is:", similarity_score)
