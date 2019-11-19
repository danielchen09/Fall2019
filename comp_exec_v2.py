import pygame
from random import randint
from paddle import Paddle
from ball import Ball
import numpy as np
import ai


class Model:
    def __init__(self, w1=None, w2=None):
        self.inputSize = 3
        self.hiddenSize = 2
        self.outputSize = 1
        self.score = 0
        self.lastPos = None

        if w1 is None or w2 is None:
            self.w1 = np.random.uniform(-0.1, 0.1, (self.hiddenSize, self.inputSize))
            self.w2 = np.random.uniform(-0.1, 0.1, (self.outputSize, self.hiddenSize))
        else:
            self.w1 = w1
            self.w2 = w2

    def getAction(self, rgb, paddleA, paddleB, ball, reward, done):
        input = np.array([paddleB.y, ball.y, ball.x]).reshape(-1, 1)
        hidden_out = self.relu(self.w1.dot(input))
        out = self.sigmoid(self.w2.dot(hidden_out))
        out = 5 * (-1 if np.random.uniform(0, 1) > out[0, 0] else 1)
        return out

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-1 * x))

    def relu(self, vector):
        vector[vector < 0] = 0
        return vector


class Game():
    def __init__(self, fA, fB, user):
        self.fA = fA  # First program's "getAction" function
        self.fB = fB  # Second program's " "
        self.user = user  # Boolean indicating whether or not to get user input
        self.draw = True
        self.maxScore = 5

        self.mutateRate = 1e-4
        self.modelSize = 5
        self.models = [Model() for i in range(self.modelSize)]
        self.current = 0
        self.epochs = 200

    def saveWeight(self, w, name):
        np.savetxt(name + ".txt", w)

    def loadWeight(self, name):
        return np.loadtxt(name + ".txt")

    def selection(self):
        return sorted(self.models, key=(lambda m: m.score), reverse=True)[:2]

    def crossOverWeight(self, w1, w2):
        mask = np.random.randint(0, 2, size=(w1.shape)).astype(np.bool)
        temp = np.array(w1[mask], copy=True)
        w1[mask] = w2[mask]
        w2[mask] = temp

        return np.array(w1, copy=True), np.array(w2, copy=True)

    def mutateWeight(self, w):
        mask = np.random.randint(0, 2, size=w.shape).astype(np.bool)
        rand = np.random.uniform(-1, 1, w.shape) * self.mutateRate
        w[mask] = rand[mask]

        return np.array(w, copy=True)

    def runComp(self):
        for epoch in range(self.epochs * self.modelSize):
            self.reset()

            pygame.init()
            pygame.display.set_caption("Pong Competition")

            last = 0
            while not self.done:
                if self.scoreB + self.scoreA != last:
                    last = self.scoreB + self.scoreA
                self.step()

            self.models[self.current].score = self.scoreB
            pygame.quit()
            print("%s wins! %d : %d" % (self.winner, self.scoreA, self.scoreB))

            self.current += 1
            if self.current == self.modelSize:
                best = self.selection()
                print("epoch %s best: %d" % (epoch // self.modelSize, best[0].score))
                m1_w1_new, m2_w1_new = self.crossOverWeight(best[0].w1, best[1].w1)
                m1_w2_new, m2_w2_new = self.crossOverWeight(best[0].w2, best[1].w2)
                self.saveWeight(m1_w1_new, "w1")
                self.saveWeight(m1_w2_new, "w2")
                print(np.mean(m1_w1_new), np.mean(m1_w2_new))
                newmodel = [Model(m1_w1_new, m1_w2_new), Model(m2_w1_new, m2_w2_new)]
                for i in range(len(newmodel), self.modelSize):
                    newmodel.append(Model(self.mutateWeight(m1_w1_new), self.mutateWeight(m1_w2_new)))
                self.models = newmodel
                self.current = 0

    def step(self):
        self.reward = 0

        # PYGAME
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                self.done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                    self.done = True

        if self.scoreA == self.maxScore or self.scoreB == self.maxScore:
            self.winner = "Player A" if self.scoreA == self.maxScore else "Player B"
            self.done = True

        # Getting screen pixels
        rgbarray = pygame.surfarray.array3d(pygame.display.get_surface())

        # Compiling useful information
        info = [rgbarray, self.paddleA.rect, self.paddleB.rect, self.ball.rect, self.reward, self.done]

        # Sending info to first function to get action
        actionA = self.fA(*info)
        self.paddleA.moveUp(actionA)

        # If indicated that user is providing input
        if self.user:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.paddleB.moveUp(10)
            if keys[pygame.K_DOWN]:
                self.paddleB.moveDown(10)
        # If two programs playing against each other
        else:
            actionB = self.models[self.current].getAction(*info)
            self.paddleB.moveUp(actionB)
            # print(actionB)

        # PYGAME
        if self.paddleB.rect.y > 600 or self.paddleB.rect.y < 100:
            self.paddleB.rect.y = 600 if self.paddleB.rect.y > 600 else 100
        if self.paddleA.rect.y > 600 or self.paddleA.rect.y < 100:
            self.paddleA.rect.y = 600 if self.paddleA.rect.y > 600 else 100

        # PYGAME
        self.all_sprites_list.update()
        if self.ball.rect.y > 685 or self.ball.rect.y < 100:
            self.ball.rect.y = 100 if self.ball.rect.y < 100 else 685
            self.ball.velocity[1] = -self.ball.velocity[1]
        if self.ball.rect.x >= 490:
            self.scoreA += 1
            self.reward = -1
            self.ball.rect.x = 250
            self.ball.rect.y = 300
            self.ball.velocity = [2 if randint(0, 1) == 0 else -2, 2 if randint(0, 1) == 0 else -2]
        if self.ball.rect.x <= 0:
            self.scoreB += 1
            self.reward = 1
            self.ball.rect.x = 250
            self.ball.rect.y = 300
            self.ball.velocity = [2 if randint(0, 1) == 0 else -2, 2 if randint(0, 1) == 0 else -2]
        # Detect collisions between the ball and the paddles
        if pygame.sprite.collide_mask(self.ball, self.paddleA) or pygame.sprite.collide_mask(self.ball, self.paddleB):
            self.ball.bounce()

        if self.draw:
            # --- Drawing code should go here
            # First, clear the screen to BLACK.
            self.screen.fill(self.BLACK)
            # Draw the net
            pygame.draw.line(self.screen, self.WHITE, [0, 100], [500, 100], 10)
            # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
            self.all_sprites_list.draw(self.screen)
            # Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.scoreA), 1, self.WHITE)
            self.screen.blit(text, (125, 10))
            text = font.render(str(self.scoreB), 1, self.WHITE)
            self.screen.blit(text, (375, 10))

        pygame.display.flip()
        return info

    def reset(self):
        self.size = (500, 700)

        self.BLACK = (144, 0, 0)
        self.WHITE = (255, 255, 255)
        self.screen = pygame.display.set_mode(self.size)
        self.paddleA = Paddle(self.WHITE, 10, 100)
        self.paddleA.rect.x = 30
        self.paddleA.rect.y = 300
        self.paddleB = Paddle(self.WHITE, 10, 100)
        self.paddleB.rect.x = 470
        self.paddleB.rect.y = 300
        self.ball = Ball(self.WHITE, 20, 20)
        self.ball.rect.x = 250
        self.ball.rect.y = 300
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.paddleA)
        self.all_sprites_list.add(self.paddleB)
        self.all_sprites_list.add(self.ball)
        self.winner = "No one yet"

        self.done = False
        self.scoreA = 0
        self.scoreB = 0

        rgbarray = pygame.surfarray.array3d(pygame.display.get_surface())
        info = [rgbarray, self.paddleA.rect, self.paddleB.rect, self.ball.rect, 0, self.done]


m = Model()
g = Game(ai.getAction, m.getAction, False)
g.runComp()  

