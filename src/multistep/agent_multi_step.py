import torch
import random
import numpy as np
from collections import deque
from snake_train import snake_game, Direction
from model import Linear_QNet, QTrainer
from helper import plot
from collections import namedtuple

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
STEP_SIZE = 3
LR = 0.001


class Agent:

    def __init__(self, snake_num=1):
        self.snake_num = snake_num
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=STEP_SIZE) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)





    def get_state(self, game):
        if self.snake_num == 1:
            head = (game.snake1_x, game.snake1_y)
            dir = game.direction1
        elif self.snake_num == 2:
            head = (game.snake2_x, game.snake2_y)
            dir = game.direction2

        block_size = game.snake_block
        point_l = (head[0] - block_size, head[1])
        point_r = (head[0] + block_size, head[1])
        point_u = (head[0], head[1] - block_size)
        point_d = (head[0], head[1] + block_size)
        
        dir_l = dir == Direction.LEFT
        dir_r = dir == Direction.RIGHT
        dir_u = dir == Direction.UP
        dir_d = dir == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._collision(point_r)) or 
            (dir_l and game._collision(point_l)) or 
            (dir_u and game._collision(point_u)) or 
            (dir_d and game._collision(point_d)),

            # Danger right
            (dir_u and game._collision(point_r)) or 
            (dir_d and game._collision(point_l)) or 
            (dir_l and game._collision(point_u)) or 
            (dir_r and game._collision(point_d)),

            # Danger left
            (dir_d and game._collision(point_r)) or 
            (dir_u and game._collision(point_l)) or 
            (dir_r and game._collision(point_u)) or 
            (dir_l and game._collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.foodx < head[0],  # food left
            game.foodx > head[0],  # food right
            game.foody < head[1],  # food up
            game.foody > head[1]  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self):
        if len(self.memory) == STEP_SIZE:
            # mini_sample = random.sample(self.memory, STEP_SIZE) # list of tuples

            states, actions, rewards, next_states, dones = zip(*self.memory)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 40
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent1 = Agent(snake_num = 1)
    game = snake_game()
    while True:
        # get old state
        state_old = agent1.get_state(game)

        # get move
        action1 = agent1.get_action(state_old)

        # perform move and get new state
        reward1, reward2, done = game.play(action1)
        state_new = agent1.get_state(game)

        # remember
        agent1.remember(state_old, action1, reward1, state_new, done)

        # train short memory
        agent1.train_short_memory()


        if done:
            # train long memory, plot result
            agent1.n_games += 1
            # agent1.train_long_memory()

            if game.score1 >= record:
                record = game.score1
                agent1.model.save(file_name='best_model.pth')

            if agent1.n_games % 200 == 0:
                agent1.model.save(file_name='latest_model.pth')

            print('Game', agent1.n_games, 'Score', game.score1, 'Record:', record)

            plot_scores.append(game.score1)
            total_score += game.score1
            mean_score = total_score / agent1.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            game.reset()
            agent1.memory.clear()


if __name__ == '__main__':
    train()