import numpy as np 

def softmax_forward(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs =  exp / np.sum(exp, axis=1, keepdims=True) 
    return probs

def cross_entropy_loss(probs, labels):
    N = probs.shape[0]
    correct_logprobs = -np.log(probs[np.arange(N), labels] + 1e-12)  
    loss = correct_logprobs.mean()
    return loss

def softmax_cross_entropy_backward(probs, labels):
    N = probs.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits = dlogits/N 
    return dlogits