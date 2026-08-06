import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)

y = y.astype(int)
X = X / 255.0

X_train , X_test = X[:60000], X[60000:]
y_train , y_test = y[:60000], y[60000:]

print("\nTrain:", X_train.shape, y_train.shape)
print("\nTest : ", X_test.shape, y_test.shape)

fig, axes = plt.subplots(2 , 5, figsize = (10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28),cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist.png")
plt.show()

np.savez('mnist_data.npz', X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("\nSaved to mnist_data.npz")

