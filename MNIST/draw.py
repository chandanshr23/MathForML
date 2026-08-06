"""
Draw a digit, get a live prediction from the trained MNIST MLP.

Run locally: python draw_predict.py
Requires: pip install pillow  (tkinter itself ships with standard Python on Windows)
"""
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from model import MLP

CANVAS_SIZE = 280       # draw big — 10x the target 28x28, easier to draw accurately
BRUSH_RADIUS = 12       # thick brush — MNIST strokes are fairly heavy, thin lines
                        # underrepresent the training distribution


class DigitDrawer:
    def __init__(self, root, model):
        self.model = model

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # a parallel PIL image we draw into alongside the visible tkinter canvas —
        # tkinter canvases aren't directly readable as pixel arrays, so we mirror
        # every stroke onto a PIL Image we CAN read pixels from
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black background
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=5)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.grid(row=1, column=1, pady=5)

        self.result_label = tk.Label(root, text="Draw a digit, then click Predict", font=("Arial", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        # draw on the VISIBLE canvas (white stroke, since bg is black — matches MNIST look)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # mirror the same stroke onto the PIL image we'll actually read pixels from
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit, then click Predict")

    def preprocess(self):
        """
        Match MNIST's format properly — not just resize, but center-of-mass
        centering, which is how the real MNIST dataset was built. Skipping this
        is the most likely reason drawn digits predict worse than the model's
        97% test accuracy suggests.
        """
        # crop to the drawn content's bounding box first — an off-center stroke
        # inside a mostly-empty 280x280 canvas confuses a naive resize
        bbox = self.image.getbbox()
        if bbox is None:
            # nothing drawn yet
            arr = np.zeros((28, 28), dtype=np.float32)
            return arr.reshape(1, 784), arr

        cropped = self.image.crop(bbox)

        # resize the cropped digit to fit in a 20x20 box (MNIST's convention),
        # preserving aspect ratio
        w, h = cropped.size
        if w > h:
            new_w, new_h = 20, max(1, int(20 * h / w))
        else:
            new_h, new_w = 20, max(1, int(20 * w / h))
        resized = cropped.resize((new_w, new_h), Image.LANCZOS)

        # paste the 20x20-ish digit into the center of a 28x28 black canvas
        canvas28 = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas28.paste(resized, (paste_x, paste_y))

        # center of mass shift — real MNIST centers by intensity-weighted
        # centroid, not just the bounding box center, so nudge further
        arr = np.array(canvas28, dtype=np.float32)
        total = arr.sum()
        if total > 0:
            ys, xs = np.indices(arr.shape)
            cy = (ys * arr).sum() / total
            cx = (xs * arr).sum() / total
            shift_y = int(round(14 - cy))
            shift_x = int(round(14 - cx))
            arr = np.roll(arr, shift_y, axis=0)
            arr = np.roll(arr, shift_x, axis=1)

        arr_norm = arr / 255.0
        flat = arr_norm.reshape(1, 784)
        return flat, arr_norm

    def predict(self):
        X, _ = self.preprocess()
        # training=False: dropout off, BatchNorm uses running stats accumulated
        # during training — this is exactly why that fix mattered earlier
        logits, _ = self.model.forward(X, training=False)
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        pred = int(np.argmax(probs))
        confidence = float(probs[0, pred])
        self.result_label.config(text=f"Prediction: {pred}   (confidence {confidence:.1%})")


if __name__ == "__main__":
    model = MLP()
    model.load("mnist_weights.npz")

    root = tk.Tk()
    root.title("Draw a digit")
    app = DigitDrawer(root, model)
    root.mainloop()