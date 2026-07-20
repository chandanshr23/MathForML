import random

# ---------- Value class (your autograd engine) ----------
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

    #To check the backward is correct we use Gradient Checking 
    def grad_check(value, variable, h=1e-6):
         # Save original value
        orig = variable.data

        # f(x+h)
        variable.data = orig + h
        plus = value().data

        # f(x-h)
        variable.data = orig - h
        minus = value().data

        # Restore
        variable.data = orig

        return (plus - minus) / (2*h)
    
    

# USer this part to the code for learning purpose
# class Value:

#     def __init__(self,data,_children=(),op=''):
#         self.data = data
#         self.grad = 0
#         self._backward = lambda:None
#         self._prev = set(_children)
#         self._op = op

#     def __repr__(self):
#         return f"Value.data = {self.data} and Value.grad= {self.grad}"
# # a = Value(3.0)
# # print(a)
# # print(a.prev)

# # b = Value(4.0)
# # print(b)
# # print(b.prev)

# #Add 
    
#     def __add__(self, other):
#         other = other if isinstance(other , Value) else Value(other)
#         out = Value(self.data + other.data , (self,other), '+')

#         def _backward():
#             self.grad += out.grad
#             other.grad += out.grad
#         out._backward = _backward
        
#         return out
    
# # a = Value(3.0)
# # b = Value(2.0)
# # c = a+b
# # c.grad = 1.0
# # c._backward()
# # print(a.grad)
# # print(b.grad)

#     def __mul__(self, other):
#         other = other if isinstance(other , Value) else Value(other)
#         out = Value(self.data * other.data, (self,other),'*')

#         def _backward():
#             self.grad =other.data * out.grad
#             other.grad = self.data * out.grad
#         out._backward = _backward

#         return out
    
# # a = Value(3.0)
# # b = Value(2.0)
# # c = a*b 
# # c.grad = 1.0
# # c._backward()
# # print(a.grad)
# # print(b.grad)
# # print(c.data)
# # print(c._prev)

# # a = Value(2.0)
# # b = Value(-3.0)
# # c = a * b          # c = -6
# # d = c + a          # d = -4   (this is your final output)

# # d.grad = 1.0
# # d._backward()       # pushes gradient into c and a (the '+' contributes 1 to each)
# # c._backward()       # pushes gradient from c into a and b (the '*' rule)

# # print(a.grad)  # should be 1 (from d's + ) + b.data=-3 (from c's *) = -2
# # print(b.grad)  # should be a.data = 2

#     def backward(self):
#         topo = []
#         visited = set()
#         def build_node(v):
#             if v is not visited:
#                 for child in v._prev:
#                     build_node(child)
#                 topo.append(v)
#             build_node(self)
#         self.grad = 1
#         for v in reversed(topo):
#             v.backward()

#     def __pow__(self, other):
#         assert isinstance(other, (int, float))
#         out = Value(self.data**other, (self,), f'**{other}')
#         def _backward():
#             self.grad += (other * self.data**(other-1)) * out.grad
#         out._backward = _backward
#         return out

#     def relu(self):
#         out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
#         def _backward():
#             self.grad += (out.data > 0) * out.grad
#         out._backward = _backward
#         return out
    
#     def __neg__(self):
#         return self * -1
    
#     def __radd__(self,other):
#         return self + other
    
#     def __sub__(self, other):
#         return self + (-other)
    
#     def __rsub__(self,other):
#         return other + (-self)
    
#     def __rmul__(self,other):
#         return self * other
    
#     def __truediv__(self, other):
#         return self * other **-1
    
#     def __rtruediv__(self , other):
#         return other * self **-1
    
#     def __repr__(self):
#         return f"Value(data={self.data} , grad ={self.grad})"
    
