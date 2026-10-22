"""Two-layer ReLU classifier with explicit vectorized backpropagation."""
import numpy as np


class MLP:
    def __init__(self, inputs=784, hidden=64, outputs=10):
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs
        self.sizes = [inputs*hidden, hidden, hidden*outputs, outputs]
        self.cuts = np.cumsum([0]+self.sizes)

    def unpack(self, theta):
        a,b,c,d = [theta[self.cuts[i]:self.cuts[i+1]] for i in range(4)]
        return a.reshape(self.inputs,self.hidden), b, c.reshape(self.hidden,self.outputs), d

    def initialize(self, seed):
        rng = np.random.default_rng(seed)
        return np.concatenate([rng.normal(0, np.sqrt(2/self.inputs), self.sizes[0]),
                               np.zeros(self.hidden),
                               rng.normal(0, np.sqrt(2/self.hidden), self.sizes[2]),
                               np.zeros(self.outputs)]).astype(np.float64)

    def loss_gradient(self, theta, x, y, gradient=True):
        w,b,u,c = self.unpack(theta)
        pre = x @ w + b
        hidden = np.maximum(pre, 0.)
        logits = hidden @ u + c
        shifted = logits-logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        logsum = np.log(exp.sum(axis=1))
        loss = float(np.mean(logsum-shifted[np.arange(len(y)), y]))
        if not gradient:
            return loss, logits.argmax(axis=1)
        delta = exp/exp.sum(axis=1,keepdims=True)
        delta[np.arange(len(y)),y] -= 1.
        delta /= len(y)
        du = hidden.T @ delta
        dc = delta.sum(axis=0)
        dh = (delta @ u.T) * (pre > 0)
        dw = x.T @ dh
        db = dh.sum(axis=0)
        return loss, np.concatenate([dw.ravel(),db,du.ravel(),dc])

    def evaluate(self, theta, x, y, chunk=512):
        total_loss = 0.
        correct = 0
        for start in range(0,len(y),chunk):
            labels = y[start:start+chunk]
            loss,pred = self.loss_gradient(theta,x[start:start+chunk],labels,False)
            total_loss += loss*len(labels)
            correct += int((pred==labels).sum())
        return dict(loss=total_loss/len(y),accuracy=correct/len(y))
