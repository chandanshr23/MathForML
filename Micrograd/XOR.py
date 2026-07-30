from micrograd import Value
from micrograd import cross_entropy
from micrograd import softmax_stable
from micrograd import softmax
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

model = MLP(2, [4, 2])   # 2 inputs, 4 hidden, 2 output logits (last layer linear automatically)

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0, 1, 1, 0]   # class index version of XOR truth table

learning_rate = 0.05

for epoch in range(200):
    total_loss = Value(0.0)
    for x, target_idx in zip(xs, ys):
        logits = model(x)
        loss = cross_entropy(logits, target_idx)
        total_loss = total_loss + loss

    model.zero_grad()
    total_loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss {total_loss.data:.4f}")
print("\nfinal predictions:")
for x, target_idx in zip(xs, ys):
    logits = model(x)
    probs = softmax(logits)
    predicted = 0 if probs[0].data > probs[1].data else 1
    print(f"input {x}, target {target_idx}, predicted {predicted}, probs {[round(p.data,3) for p in probs]}")
print("\nnaive softmax on VERY large logits:")
huge_logits = [Value(800.0), Value(900.0), Value(1000.0)]
probs = softmax(huge_logits)          # <-- comment out or delete this line
print([p.data for p in probs])        # <-- and this one

print("\nstable softmax on same VERY large logits:")
huge_logits2 = [Value(800.0), Value(900.0), Value(1000.0)]
probs_stable = softmax_stable(huge_logits2)
print([p.data for p in probs_stable])
print(sum(p.data for p in probs_stable))