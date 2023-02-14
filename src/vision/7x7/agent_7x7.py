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
STEP_SIZE = 5
LR = 0.001


class Agent:

    def __init__(self, snake_num=1):
        self.vision_size = 7
        self.snake_num = snake_num
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(49 + 8, 256, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        if self.snake_num == 1:
            head = (game.snake1_x, game.snake1_y)
            dir = game.direction1
        elif self.snake_num == 2:
            head = (game.snake2_x, game.snake2_y)
            dir = game.direction2

        # vision = game.get_snake_vision(self.snake_num)

        # print(vision)
        # vision = vision.flatten().tolist()

        
        block_size = game.snake_block
        point_l = (head[0] - block_size, head[1])
        point_r = (head[0] + block_size, head[1])
        point_u = (head[0], head[1] - block_size)
        point_d = (head[0], head[1] + block_size)
        
        point = np.zeros((self.vision_size, self.vision_size), int)
        for i in range(self.vision_size):
            for j in range(self.vision_size):
                if game._collision((head[0] + block_size * (j - (self.vision_size-1)/2), head[1] + block_size * (i - (self.vision_size-1)/2))):
                    point[i][j] = 1
                # elif game._is_food((head[0] + block_size * (j - 1), head[1] + block_size * (i - 1))):
                #     point[i][j] = 1
                else:
                    point[i][j] = 0
                

        if dir == Direction.LEFT:
            # print("LEFT")
            point = np.rot90(point, k = -1)
        elif dir == Direction.DOWN:
            # print("DOWN")
            point = np.rot90(point, k = 2)
        elif dir == Direction.RIGHT:
            # print("RIGHT")
            point = np.rot90(point, k = 1)
        # else:
            # print("UP")
        # print(point)

        dir_l = dir == Direction.LEFT
        dir_r = dir == Direction.RIGHT
        dir_u = dir == Direction.UP
        dir_d = dir == Direction.DOWN

        

        state = [
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

        state = point.flatten().tolist() + state

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

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 30 - (20 if self.n_games > 2000 else self.n_games/100)
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

history = deque(maxlen=50) # popleft()
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record1 = 0
    record2 = 0
    agent1 = Agent(snake_num = 1)
    # agent2 = Agent(snake_num = 2)
    game = snake_game()
    while True:
        # get old state
        state_old1 = agent1.get_state(game)

        # get move
        action1 = agent1.get_action(state_old1)

        # get old state
        # state_old2 = agent2.get_state(game)

        # get move
        # action2 = agent2.get_action(state_old2)



        # perform move and get new state
        reward1, reward2, done = game.play(action1)

        state_new1 = agent1.get_state(game)

        # train short memory
        agent1.train_short_memory(state_old1, action1, reward1, state_new1, done)
        
        # remember
        agent1.remember(state_old1, action1, reward1, state_new1, done)


        # state_new2 = agent2.get_state(game)

        # train short memory
        # agent2.train_short_memory(state_old2, action2, reward2, state_new2, done)
        
        # remember
        # agent2.remember(state_old2, action2, reward2, state_new2, done)

        if done == True:
            # train long memory, plot result
            agent1.n_games += 1
            # agent2.n_games += 1
            agent1.train_long_memory()
            # agent2.train_long_memory()

            if game.score1 >= record1:
                record1 = game.score1
                # agent1.model.save(file_name='snake1_best_model.pth')

            if game.score2 >= record2:
                record2 = game.score2
                # agent2.model.save(file_name='snake2_best_model.pth')

            if game.score2 > record2:
                record2 = game.score2
                # agent2.model.save()

            print('Game', agent1.n_games, 'Score1', game.score1, 'Score2', game.score2, 'Record1:', record1, 'Record2:', record2)

            plot_scores.append(game.score1)
            # total_score += game.score1
            history.append(game.score1)
            total_score = 0
            for i in history:
                total_score += i
            mean_score = total_score / len(history)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            # if agent1.n_games > 2000:
            #     quit()
            game.reset()
            if agent1.n_games > 8000:
                quit()


if __name__ == '__main__':
    train()