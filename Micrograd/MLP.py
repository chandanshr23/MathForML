#Refer the below code for the implementation of MLP and to understand comment out this part of code and uncomment the next part of the code 

import random
from micrograd import Value
import numpy as np


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"









# from micrograd import Value
# import numpy as np


# class Module:

#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = 0

#     def parameters(self):
#         return []
    
# class Neuron(Module):

# #To Check manually for hard coded values : 
#     # def __init__(self,nin,nonLin=True):
#     #     self.w = [Value(1.0) for _ in range(nin)]
#     #     self.b = Value(0.0)
#     #     self.nonLin = nonLin

#     # def __call__(self, x) :
#     #     act = sum((wi*xi for wi,xi in zip(self.w , x)),self.b)
#     #     return act.relu() if self.nonLin else act 
    
#     # def parameter(self):
#     #     return self.w + [self.b]
# # n = Neuron(2)
# # out = n([2.0 ,3.0])
# # print(out.data)

# # n = Neuron(2)
# # out = n([-10.0, 2.0])
# # print(out.data)   

# # n = Neuron(2, nonLin=False)
# # out = n([-10.0, 2.0])
# # print(out.data)   # -8, NOT clipped to 0

# # n = Neuron(3)
# # print(len(n.parameters()))   # 3 weights + 1 bias = 4


# class Layer(Module):
#     def __init__(self, nin ,nout, **kwargs ):
#         self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

#     def __call__(self, x):
#         out = [n(x) for n in self.neurons]
#         return out[0] if len(out) == 1 else out

#     def parameters(self):
#         return [p for n in self.neurons for p in n.parameters()] 
    
# # l = Layer(2, 1)
# # out = l([2.0, 3.0])
# # print(out)
# # print(type(out))

# # l = Layer(2, 3)
# # out = l([2.0, 3.0])
# # print(len(out))   

# # l = Layer(2, 3)
# # print(len(l.parameters()))  

# class MLP(Module):
#     def __init__(self,nin,nouts):
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1], nonLin = i!=len(nouts)-1)
#                         for i in range(len(nouts))]
          
#         def __call__(self, x):
#             for layer in self.layers:
#                 x = layer(x)
#             return x
        
#         def parameters(self):
#             return [p for layer in self.layers for p in layer.parameters()]
        
# n = MLP(2,[2,1])
# print(len(n.layers))
# print(n.layers[0].neurons[0].nonLin)
# print(n.layers[1].neurons[0].nonLin)

# print(len(n.parameters()))
# # layer1: 2 neurons * (2w+1b) = 6
# # layer2: 1 neuron  * (2w+1b) = 3
# # total = 9