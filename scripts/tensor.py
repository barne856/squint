import numpy as np

# Create input tensors A and B
A = np.array([[[1, 3], [2, 4]],
              [[5, 7], [6, 8]]])

B = np.array([[[1, 3], [2, 4]],
              [[5, 7], [6, 8]]])

# Perform the contraction using einsum
# The contraction pair is {1, 0}, so we contract the second index of A with the first index of B
result = np.einsum('ijk,lmj->ikml', A, B)

# Print the result
print(result)