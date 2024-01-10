import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardizing the data
scaler = StandardScaler()
# Original data
data = np.array([
    [8000, 7, 2200],
    [6000, 6, 2000],
    [10000, 8, 2500],
    [7500, 7, 2300],
    [9000, 6, 2400]
])
data = scaler.fit_transform(data)

# Calculating the covariance matrix
covariance_matrix = np.cov(data.T)

# Calculating eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Normalize eigenvectors (Principal Components)
norm_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Projecting the data onto the principal components
# Fit every variable , so using entire data
PC1 = data.dot(norm_eigenvectors[:, 0])
PC2 = data.dot(norm_eigenvectors[:, 1])
PC3 = data.dot(norm_eigenvectors[:, 2])

# Absolute values of projections on PC1, PC2, and PC3
abs_PC1 = np.abs(PC1)
abs_PC2 = np.abs(PC2)
abs_PC3 = np.abs(PC3)

print("Standardized data")
print(data)

print("Eigen values")
print(eigenvalues)

print("unit eigen vector")
print(norm_eigenvectors)

print("PCA")
print(abs_PC1)
print(abs_PC2)
print(abs_PC3)