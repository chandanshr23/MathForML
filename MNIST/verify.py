"""
Run locally, after training. Two checks:
1. Confirm the model really gets ~97% on actual MNIST test images (sanity check
   that the model/weights are fine).
2. Show what a REAL MNIST test image looks like after your resize pipeline,
   compared to what it looked like originally — helps you see if your drawn
   digit's preprocessing is producing something visually different.
"""
import numpy as np
import matplotlib.pyplot as plt
from model import MLP

data = np.load('mnist_data.npz')
X_test = data["X_test"]
y_test = data["y_test"]

model = MLP()
model.load("mnist_weights.npz")

logits, _ = model.forward(X_test, training=False)
preds = np.argmax(logits, axis=1)
acc = np.mean(preds == y_test)
print(f"Real MNIST test accuracy: {acc:.4f}")

# show a handful of real test digits + the model's prediction for each,
# so you can visually compare their centering/thickness against your drawings
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    correct = "OK" if preds[i] == y_test[i] else "WRONG"
    ax.set_title(f"true={y_test[i]} pred={preds[i]} [{correct}]")
    ax.axis("off")
plt.tight_layout()
plt.savefig("test_predictions.png")
print("Saved test_predictions.png — look at how centered/thick these digits are")