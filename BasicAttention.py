import numpy as np

def softmax(x, axis=0):
    """ Calculate softmax function for an array x along specified axis
    
        axis=0 calculates softmax across rows which means each column sums to 1 
        axis=1 calculates softmax across columns which means each row sums to 1
    """
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)

hidden_size = 16
attention_size = 10
input_length = 5

np.random.seed(42)

# Synthetic vectors used to test
encoder_states = np.random.randn(input_length, hidden_size)
decoder_state = np.random.randn(1, hidden_size)

# Weights for the neural network, these are typically learned through training
# Use these in the alignment function below as the layer weights
layer_1 = np.random.randn(2 * hidden_size, attention_size)
layer_2 = np.random.randn(attention_size, 1)

def alignment(encoder_states, decoder_state):
    # First, concatenate the encoder states and the decoder state.
    inputs = np.concatenate((encoder_states, decoder_state.repeat(input_length, axis=0)), axis=1)
    assert inputs.shape == (input_length, 2 * hidden_size)
    
    # Matrix multiplication of the concatenated inputs and the first layer, with tanh activation
    activations = np.tanh(np.matmul(inputs, layer_1))
    assert activations.shape == (input_length, attention_size)
    
    # Matrix multiplication of the activations with the second layer. Remember that you don't need tanh here
    scores = np.matmul(activations, layer_2)
    assert scores.shape == (input_length, 1)
    
    return scores

def attention(encoder_states, decoder_state):
    """ Example function that calculates attention, returns the context vector 
    
        Arguments:
        encoder_vectors: NxM numpy array, where N is the number of vectors and M is the vector length
        decoder_vector: 1xM numpy array, M is the vector length, much be the same M as encoder_vectors
    """ 
    
    # First, calculate the alignment scores
    scores = alignment(encoder_states, decoder_state)
    
    # Then take the softmax of the alignment scores to get a weight distribution
    weights = softmax(scores, axis=0)
    
    # Multiply each encoder state by its respective weight
    weighted_scores = encoder_states * weights
    
    # Sum up weighted alignment vectors to get the context vector and return it
    context = np.sum(weighted_scores, axis=0)
    return context

# Run this to test your alignment function
scores = alignment(encoder_states, decoder_state)
print("Alignment Scores:\n", scores)

# Run this to test your attention function
context_vector = attention(encoder_states, decoder_state)
print("Context Vector:\n", context_vector)
