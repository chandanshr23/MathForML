import math

class Value:
    def __init__(self,data, _children=(),_op=''):
        self.data = data 
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op


    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.grad * out.grad
        out._backward = _backward

    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1/x) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        return Value(1.0) / (Value(1.0) + (-self).exp())
    
    def log_loss(pred, y_true):
    # pred is a Value (probability, output of sigmoid)
    # y_true is a plain number, 0 or 1
        if y_true == 1:
            return -pred.log()
        else:
            return -(Value(1.0) - pred).log()
        

w = Value(1.0)
x = Value(3.0)
y_true = 1

# forward pass
logit = w * x
p = logit.sigmoid()
loss = log_loss(p, y_true)

# backward pass
loss.backward()   # this walks the graph in reverse, calling each _backward()

print(w.grad)   # should be some negative number, telling you to increase w

# optimizer step
lr = 0.1
w.data -= lr * w.grad