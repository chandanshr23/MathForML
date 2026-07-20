from micrograd import Value
from MLP import MLP
import random 
random.seed(127)

n = MLP(2,[4,1])

xs = [
    [0.0 , 0.0],
    [0.0 , 1.0],
    [1.0 , 0.0],
    [1.0 , 1.0]
]

ys = [0.0, 1.0,1.0, 0.0]

for step in range(300):
    ypred = [n(x) for x in xs ]

    loss = sum((yout - ygt )**2 for ygt, yout in zip(ys, ypred))

    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    learning_rate = 0.1

    for p in n.parameters():
        p.data -=learning_rate * p.grad

    if step % 30 == 0:
        print(f"step {step : 3d} loss = {loss.data:.4f}")

print("\n final predictionns:")
for x, y in zip(xs ,ys):
    pred = n(x).data
    print(f"input {x} taraget = {y} predicted = {pred : .4f}")

