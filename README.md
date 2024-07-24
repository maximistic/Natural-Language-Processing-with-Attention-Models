# Natural-Language-Processing-with-Attention-Models
Working with NTMs along with attention models



##Scaled Dot-Product Attention.py
###Libraries and Resources Used:
  - pickle : for loading the word-to-integer mapping dictionaries
  - matplotlib.pyplot : for visualization
  - numpy : for numerical operations and handling embeddings
  - en_words : word-to integer mapping for english
  - fr_words : word-to integer mapping for french
###Functions used:
  - tokenize : Converts a sentence into a list of integers based on the token_mapping (Unknown words are mapped to -1)
  - embed : Converts a list of tokens into their corresponding embeddings (Unknown tokens (-1) are represented as zero vectors
  - softmax : Computes the softmax of the given array (normalize the values such that they sum up to 1)
  - calculate_weights : computes the *scaled dot product attention weights*. calculates the dot product of queries and keys, scales them by the square root of the key dimensions and applies softmax. (Formula = softmax(QK^T/sqrt(dk)) * V)
  - attention_qkv : computes the attention output by multiplying the attention weights with the values
