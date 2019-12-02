import numpy as np
class Player:
    def __init__(self, w1=None, w2=None, b1=None, b2=None):
        self.inputSize = 8
        self.hiddenSize = 4
        self.outputSize = 1
        self.score = 0
        self.lastPos = None

        w1 = self.loadWeight("w1_5")
        w2 = self.loadWeight("w2_5").reshape(1, -1)
        b1 = self.loadWeight("b1_5").reshape(-1, 1)
        b2 = self.loadWeight("b2_5").reshape(-1, 1)


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

    def loadWeight(self, name):
        return np.loadtxt(name + ".txt")