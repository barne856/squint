import numpy as np

def complex_tensor_contraction(A, B):
    """
    Perform tensor contraction C[i,j,m,n] = sum(A[i,k,l,m] * B[j,l,k,n]) over k and l.
    
    Args:
    A: 4D numpy array of shape (3, 2, 2, 3)
    B: 4D numpy array of shape (3, 2, 2, 3)
    
    Returns:
    C: 4D numpy array of shape (3, 3, 3, 3)
    """
    # Ensure input tensors have the correct shape
    # assert A.shape == (3, 2, 2, 3) and B.shape == (3, 2, 2, 3), "Input tensors must have shape (3, 2, 2, 3)"
    
    # Perform the contraction
    C = np.einsum('iklm,jlkn->ijmn', A, B)
    
    return C

A = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print(A)
print(A[0][1][0])
print(A[0,1,0])

# # Create two 4D tensors with the correct shape
# A = np.zeros((3, 2, 3, 2), dtype=np.float32)
# B = np.zeros((3, 2, 3, 2), dtype=np.float32)
# 
# # Fill tensors with values
# for i in range(36):
#     # Calculate indices
#     w, x, y, z = np.unravel_index(i, (3, 2, 2, 3), order='F')
#     A[w, x, z, y] = float(i)
#     B[w, x, z, y] = float(i + 36)
# 
# print("Tensor A:")
# print(A)
# print("\nTensor B:")
# print(B)

# result = complex_tensor_contraction(A, B)
# print("NumPy Result shape:", result.shape)
# print("NumPy Result:")
# print(result)