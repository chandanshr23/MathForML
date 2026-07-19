class Value:

    def __init__(self,data,_children=(),op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = op

    def __repr__(self):
        return f"Value.data = {self.data} and Value.grad= {self.grad}"
# a = Value(3.0)
# print(a)
# print(a.prev)

# b = Value(4.0)
# print(b)
# print(b.prev)

#Add 
    
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , (self,other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
# a = Value(3.0)
# b = Value(2.0)
# c = a+b
# c.grad = 1.0
# c._backward()
# print(a.grad)
# print(b.grad)

    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data, (self,other),'*')

        def _backward():
            self.grad =other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out
    
# a = Value(3.0)
# b = Value(2.0)
# c = a*b 
# c.grad = 1.0
# c._backward()
# print(a.grad)
# print(b.grad)
# print(c.data)
# print(c._prev)

# a = Value(2.0)
# b = Value(-3.0)
# c = a * b          # c = -6
# d = c + a          # d = -4   (this is your final output)

# d.grad = 1.0
# d._backward()       # pushes gradient into c and a (the '+' contributes 1 to each)
# c._backward()       # pushes gradient from c into a and b (the '*' rule)

# print(a.grad)  # should be 1 (from d's + ) + b.data=-3 (from c's *) = -2
# print(b.grad)  # should be a.data = 2

    def backward(self):
        topo = []
        visited = set()
        def build_node(v):
            if v is not visited:
                for child in v._prev:
                    build_node(child)
                topo.append(v)
            build_node(self)
        self.grad = 1
        for v in reversed(topo):
            v.backward()