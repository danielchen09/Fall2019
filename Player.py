import numpy as np
class Player:
    def __init__(self, w1=None, w2=None, b1=None, b2=None):
        self.inputSize = 8
        self.hiddenSize = 4
        self.outputSize = 1
        self.score = 0
        self.lastPos = None

        if w1 is None or w2 is None:
            self.w1 = np.random.uniform(-1e-4, 1e-4, (self.hiddenSize, self.inputSize))
            self.w2 = np.random.uniform(-1e-4, 1e-4, (self.outputSize, self.hiddenSize))
        else:
            self.w1 = w1
            self.w2 = w2

        if b1 is None or b2 is None:
            self.b1 = np.random.uniform(-1e-7, 1e-7, (self.hiddenSize, 1))
            self.b2 = np.random.uniform(-1e-7, 1e-7, (self.outputSize, 1))
        else:
            self.b1 = b1
            self.b2 = b2


    def getAction(self, rgb, paddleA, paddleB, ball, reward, done):
        if self.lastPos is None:
            self.lastPos = np.array([paddleA.y, paddleB.y, ball.y, ball.x])
            return 0
        pos = np.array([paddleA.y, paddleB.y, ball.y, ball.x])
        input = np.concatenate((pos - self.lastPos, pos)).reshape(-1, 1)
        hidden_out = self.relu(self.w1.dot(input) + self.b1)
        out = self.sigmoid(self.w2.dot(hidden_out) + self.b2)
        out = 7 * (-1 if out[0, 0] < 0.5 else 1)
        return out

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-1 * x))

    def relu(self, vector):
        vector[vector < 0] = vector[vector < 0] * 0.2
        return vector