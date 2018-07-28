#you can't name file as numpy.py
#it will shadow the real numpy module

import numpy as np
from scipy import sparse

x=np.array([[1,2,3],[4,5,6]])
print("x:\n{}".format(x))

eye=np.eye(4)
print("NumPy array:\n{}".format(eye))

sparse_matrix=sparse.csr_matrix(eye)
print("SciPy sparse CSR matri:\n{}".format(sparse_matrix))

data=np.ones(4)
row_indices=np.arange(4)
col_indices=np.arange(4)
eye_co=sparse.coo_matrix((data,(row_indices,col_indices)))
print("COO representation:\n{}".format(eye_co))