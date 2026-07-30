import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

p = np.array([0.4, 0.3, 0.3])
q = np.array([0.2, 0.5, 0.3])
print("KL(p,q): ", kl_divergence(p,q))
print("KL(q,p):", kl_divergence(q, p))


def entropy(p):
    return -np.sum(p * np.log(p))

def perplexity(p):
    return np.exp(entropy(p))

confident = np.array([0.97, 0.01, 0.01, 0.01])
uniform = np.array([0.25, 0.25, 0.25, 0.25 ])

print("confident : entropy", entropy(confident), "perplexity= ", perplexity(confident))
print("uniform : entropy = ", entropy(uniform), "perplexity = ", perplexity(uniform))
