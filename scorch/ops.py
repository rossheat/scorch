import numpy as np
from .tensor import *

class Op: pass

class Dot(Op):

    def forward(self, a: Tensor, w: Tensor) -> Tensor:
        self.inputs = a, w
        o = a.data.dot(w.data)
        return Tensor(o, parents=[a, w], op=self)

    def backward(self, grad):
        a, w = self.inputs
        a.backward(grad.dot(w.data.T))
        w.backward(a.data.T.dot(grad))

class ReLU(Op):
    def __call__(self, z):
        return self.forward(z)

    def forward(self, z: Tensor) -> Tensor:
        self.input = z
        a = np.maximum(0, z.data)
        return Tensor(a, parents=[z], op=self)

    def backward(self, grad):
        z = self.input
        z.backward(grad * (z.data > 0))


class AddBias(Op):
    def forward(self, o: Tensor, b: Tensor) -> Tensor:
        self.inputs = o, b
        z = o.data + b.data
        return Tensor(z, parents=[o, b], op=self)

    def backward(self, grad):
        o, b = self.inputs
        o.backward(grad)
        b.backward(np.sum(grad, axis=0, keepdims=True))


class MSELoss(Op):
    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)

    # TODO: update shape of y/y_pred to follow pytorch convention.
    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        self.inputs = y_pred, y
        loss = np.mean((y_pred.data.flatten() - y.data) ** 2)
        return Tensor(loss, parents=[y_pred, y], op=self)

    # We dont use the gradient (at this time, anyway)
    def backward(self, _):
        y_pred, y = self.inputs
        grad = 2 * (y_pred.data.flatten() - y.data) / y_pred.data.size
        # TODO: Remove this? when we switch to pytorch y/y_pred shapes
        grad = grad[:, np.newaxis]
        y_pred.backward(grad)


class CrossEntropyLoss(Op):
    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)

    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        self.inputs = y_pred, y
        # Applying softmax
        exps = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True))
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        # Computing negative log likelihood loss
        targets_one_hot = np.eye(y_pred.data.shape[1])[y.data]
        loss = (
            -np.sum(targets_one_hot * np.log(softmax_output + 1e-15))
            / y_pred.data.shape[0]
        )
        return Tensor(loss, parents=[y_pred, y], op=self)

    def backward(self, _):
        y_pred, y = self.inputs

        # Stabilize the softmax computation
        max_vals = np.max(y_pred.data, axis=1, keepdims=True)
        stabilized_logits = y_pred.data - max_vals

        # Compute softmax (TODO: reuse from forward?)
        exps = np.exp(stabilized_logits)
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        # Calculate gradient
        targets_one_hot = np.eye(y_pred.data.shape[1])[y.data]
        grad = (softmax_output - targets_one_hot) / y_pred.data.shape[0]

        # Backpropagate the gradient
        y_pred.backward(grad)


class Flatten(Op):
    def __call__(self, x: Tensor, start_dim: int = 1) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be scorch.Tensor")
        shape = x.data.shape
        flattened_shape = shape[:start_dim] + (-1,)
        x.data = np.reshape(x.data, flattened_shape)
        return x
