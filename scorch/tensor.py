from __future__ import annotations
import numpy as np

class Tensor: 
    def __init__(self, data, grad=0, parents=None, op=None):
        self.data: np.ndarray = data
        self.grad: float = grad
        self.parents: list[Tensor] = parents
        self.op = op

    def __len__(self):
        return len(self.data)

    def __eq__(self, other: Tensor):
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.data == other.data
    
    def __repr__(self) -> str:
        return f"Tensor({self.data.shape}) ({self.data})"

    @property
    def shape(self): 
        return self.data.shape

    def backward(self, grad=None):
        if grad is None: 
            grad = 1
        self.grad += grad

        if self.op: 
            self.op.backward(grad)

    def argmax(self, axis: int):
        return Tensor(np.argmax(self.data, axis=axis))

    def item(self):
        if self.data.size != 1:
            raise ValueError("Only single element tensors can be converted to Python scalars")
        return self.data.item()

    def sum(self, axis=None, keep_dims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keep_dims))