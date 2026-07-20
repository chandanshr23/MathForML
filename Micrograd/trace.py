from micrograd import Value
from MLP import MLP

print("=" * 70)
print("BUILDING THE NETWORK: MLP(2, [2, 1])")
print("  -> 2 inputs -> hidden layer of 2 neurons (ReLU) -> 1 output neuron (linear)")
print("=" * 70)
n = MLP(2, [2, 1])

print("\nWeights assigned (fixed, not random, so we can predict everything by hand):")
for li, layer in enumerate(n.layers):
    for ni, neuron in enumerate(layer.neurons):
        print(f"  Layer {li} Neuron {ni}: w={[round(w.data,3) for w in neuron.w]} b={neuron.b.data}")

x = [1.0, 2.0]
target = 0.0

print("\n" + "=" * 70)
print(f"FORWARD PASS  --  input x = {x}")
print("=" * 70)
out = n(x)
print(f"\n>>> FINAL NETWORK OUTPUT = {out.data:.4f}")

print("\n" + "=" * 70)
print("LOSS COMPUTATION")
print("=" * 70)
loss = (out - target) ** 2
print(f"loss = (output - target)^2 = ({out.data:.4f} - {target})^2 = {loss.data:.4f}")

print("\n" + "=" * 70)
print("BACKWARD PASS  --  loss.backward() builds topo order, then walks it in reverse")
print("=" * 70)
loss.backward()

print("\nGradients after backward() -- these tell us how much each weight/bias")
print("would need to change to reduce the loss:")
for li, layer in enumerate(n.layers):
    for ni, neuron in enumerate(layer.neurons):
        for wi, w in enumerate(neuron.w):
            print(f"  Layer {li} Neuron {ni} w[{wi}].grad = {w.grad:.4f}")
        print(f"  Layer {li} Neuron {ni} b.grad    = {neuron.b.grad:.4f}")

print("\n" + "=" * 70)
print("DONE -- this is exactly one iteration of what the XOR training loop does,")
print("300 times in a row, with random weights instead of fixed ones.")
print("=" * 70)