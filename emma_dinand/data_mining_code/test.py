from sklearn.decomposition import PCA
import numpy as np

# 6 vectors, each with 4 features
X = np.random.rand(6, 4)

# PCA with all possible components (max = 4 in this case)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
