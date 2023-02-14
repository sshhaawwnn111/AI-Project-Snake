import pygame
import time
import random
from enum import Enum
import numpy as np


# pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class snake_game:

    def __init__(self, w=320, h=240): # initialize pygame envrionment
       

        # boundary w/h

        self.dis_width = w
        self.dis_height = h
    
        # self.clock = pygame.time.Clock()
        self.frame_iteration = 0 # if iteration larger than a constant, terminate the game and reset (avoid infinite loop)

        # initial place of snake 1 and snake 2 and food // should be random
        self.snake_block = 10

        self.snake1_x = random.randint(0, self.dis_width//10 - self.snake_block//10) * 10
        self.snake1_y = random.randint(0, self.dis_height//10 - self.snake_block//10) * 10

        self.snake2_x = random.randint(0, self.dis_width//10 - self.snake_block//10) * 10
        self.snake2_y = random.randint(0, self.dis_height//10 - self.snake_block//10) * 10
        
        self.score1 = 0
        self.score2 = 0

        # the size of a block

        self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0

        # the initial direction of the snakes

        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.RIGHT

        self.vision_size = 1

        # speed of snake 
        self.snake1_list = []
        self.snake2_list = []

        self.snake_speed = 100
        self.snake1_length = 1
        self.snake2_length = 1

        # self.dis=pygame.display.set_mode((self.dis_width,self.dis_height))
        # pygame.display.set_caption("Snake game")

    def get_snake_vision(self, snake_num):
        # if self.snake1_x >= self.dis_width or self.snake1_x < 0 or self.snake1_y >= self.dis_height or self.snake1_y < 0:
        #     return None, None

        # if self.snake2_x >= self.dis_width or self.snake2_x < 0 or self.snake2_y >= self.dis_height or self.snake2_y < 0:
        #     return None, None 
        snake_map = np.zeros((self.dis_width//self.snake_block, self.dis_height//self.snake_block))
        snake_map[int(self.foodx)//self.snake_block, int(self.foody)//self.snake_block] = -1
        for x in self.snake1_list:
            snake_map[x[0]//10][x[1]//10] = 1
        snake_map[int(self.snake1_x//10)][int(self.snake1_y//10)] = 1 # snake1's head
        for x in self.snake2_list:
            snake_map[x[0]//10][x[1]//10] = 1
        snake_map[int(self.snake2_x//10)][int(self.snake2_y//10)] = 1 # snake2's head
        if snake_num == 1:

            # 0: ground, -1: out of bound, 1: snake1's body, 2: snake2's body, 3: snake1's head, 4: snake2's head
            vision = np.full((self.vision_size*2+1, self.vision_size*2+1), 1)
            x = 0
            for i in range(self.snake1_x//10-self.vision_size,self.snake1_x//10+self.vision_size+1):
                y = 0
                if i >= self.dis_width//self.snake_block or i<0:
                    x += 1
                    continue
                for j in range(self.snake1_y//10-self.vision_size,self.snake1_y//10+self.vision_size+1):
                    if j >= self.dis_height//self.snake_block or j<0:
                        y += 1
                        continue
                    vision[x][y] = snake_map[i][j]
                    y += 1
                x += 1
        

        if snake_num == 2:

            vision = np.full((self.vision_size*2+1, self.vision_size*2+1), -1)
            x = 0
            for i in range(self.snake2_x//10-self.vision_size,self.snake2_x//10+self.vision_size+1):
                y = 0
                if i >= self.dis_width//self.snake_block or i<0:
                    x += 1
                    continue
                for j in range(self.snake2_y//10-self.vision_size,self.snake2_y//10+self.vision_size+1):
                    if j >= self.dis_height//self.snake_block or j<0:
                        y += 1
                        continue
                    vision[x][y] = snake_map[i][j]
                    y += 1
                x += 1
        return vision

    def reset(self):
        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.RIGHT

        self.snake1_length = 1
        self.snake2_length = 1
        self.snake1_list = []
        self.snake2_list = []

        self.snake1_x = random.randint(0, self.dis_width//10 - self.snake_block//10) * 10
        self.snake1_y = random.randint(0, self.dis_height//10 - self.snake_block//10) * 10

        self.snake2_x = random.randint(0, self.dis_width//10 - self.snake_block//10) * 10
        self.snake2_y = random.randint(0, self.dis_height//10 - self.snake_block//10) * 10

        self.score1 = 0
        self.score2 = 0
        self.frame_iteration = 0

        self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
        # self.clock = pygame.time.Clock()

    # def message(self, msg, color): # function to print message
        # font_style = pygame.font.SysFont(None, 50)
        # mesg = font_style.render(msg, True, color)
        # self.dis.blit(mesg, [self.dis_width/2-75, self.dis_height/2-30])

    def _move(self, action1, action2=None):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction1)

        if np.array_equal(action1, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action1, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction1 = new_dir

        idx = clock_wise.index(self.direction2)
        if np.array_equal(action2, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action2, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction2 = new_dir

        if self.direction1 == Direction.RIGHT:
            self.snake1_x += self.snake_block
        elif self.direction1 == Direction.LEFT:
            self.snake1_x -= self.snake_block
        elif self.direction1 == Direction.DOWN:
            self.snake1_y += self.snake_block
        elif self.direction1 == Direction.UP:
            self.snake1_y -= self.snake_block
        if action2 is not None:
            if self.direction2 == Direction.RIGHT:
                self.snake2_x += self.snake_block
            elif self.direction2 == Direction.LEFT:
                self.snake2_x -= self.snake_block
            elif self.direction2 == Direction.DOWN:
                self.snake2_y += self.snake_block
            elif self.direction2 == Direction.UP:
                self.snake2_y -= self.snake_block

    def _boundary(self):
        reward1 = 0
        reward2 = 0
        if self.snake1_x >= self.dis_width or self.snake1_x < 0 or self.snake1_y >= self.dis_height or self.snake1_y < 0:
            reward1 = -10
        
        if self.snake2_x >= self.dis_width or self.snake2_x < 0 or self.snake2_y >= self.dis_height or self.snake2_y < 0:
            reward2 = -10

        return reward1, reward2

    def _collision(self, next_position=None): # bump into others or suicide, next position: (x, y)
        reward1 = 0
        reward2 = 0
        if next_position is None:
            for x in self.snake2_list:
                if x == self.snake1_list[-1]: # snake1's head
                    reward1 = -10
            for x in self.snake1_list:
                if x == self.snake2_list[-1]: # snake2's head
                    reward2 = -10

            # suicide
            for x in self.snake1_list[:-1]:
                if x == (self.snake1_x, self.snake1_y):
                    reward1 = -10
            for x in self.snake2_list[:-1]:
                if x == (self.snake2_x, self.snake2_y):
                    reward2 = -10
            return reward1, reward2

        else:
            for x in self.snake2_list:
                if x == next_position: 
                    return True
            for x in self.snake1_list:
                if x == next_position: 
                    return True
            if next_position[0] >= self.dis_width or next_position[0] < 0 or next_position[1] >= self.dis_height or next_position[1] < 0: # out of bound
                return True

            return False

    def _is_food(self, next_position=None):
        if next_position[0] == self.foodx and next_position[1] == self.foody:
            return True
        else:
            return False


    def _found_food(self):
        if self.snake1_x == self.foodx and self.snake1_y == self.foody:
            self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
            self.snake1_length += 1
            self.score1 += 1
            return 1
        elif self.snake2_x == self.foodx and self.snake2_y == self.foody:
            self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
            self.snake2_length += 1
            self.score2 += 1
            return 2
        return 0
    
    def play(self, action1, action2=None):
        white=(255, 255, 255)
        black=(0, 0, 0)
        blue=(0, 0, 255)
        green = (0, 255, 0)
        yellow = (255, 255, 102)

        reward1 = 0
        reward2 = 0
        game_over = False
    
        self.frame_iteration += 1

        # if self.frame_iteration > 100*max(self.snake1_length, self.snake2_length): 
        #     game_over = True

        if self.frame_iteration > 500: 
            game_over = True


        

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         game_over = True
        #         pygame.quit()
        #         quit()
            

        self._move(action1, action2)

        # touch the boundary
        reward1, reward2 = self._boundary()
        if reward1 != 0 or reward2 != 0:
            game_over = True
            return reward1, reward2, game_over
        

        # draw background, snake movement

        # self.dis.fill(white)
        # pygame.draw.rect(self.dis, black, [self.foodx, self.foody, self.snake_block, self.snake_block])
        self.snake1_list.append((self.snake1_x, self.snake1_y))
        if len(self.snake1_list) > self.snake1_length:
            del self.snake1_list[0]
        
        self.snake2_list.append((self.snake2_x, self.snake2_y))
        if len(self.snake2_list) > self.snake2_length:
            del self.snake2_list[0]

        # _collision
        reward1, reward2 = self._collision()
        if reward1 != 0 or reward2 != 0:
            game_over = True
            return reward1, reward2, game_over

        # draw the snake

        # for x in self.snake1_list:
            #print(self.snake_length)
            # pygame.draw.rect(self.dis, blue, [x[0], x[1], self.snake_block, self.snake_block])

        # for x in self.snake2_list:
            #print(self.snake_length)
            # pygame.draw.rect(self.dis, green, [x[0], x[1], self.snake_block, self.snake_block])

        # if snake eats the food -> generate a food position randomly

        # pygame.display.update() 
        # self.clock.tick(self.snake_speed)

        food = self._found_food()
        if food == 1 and not game_over:
            self.frame_iteration = 0
            reward1 = 10
        elif food == 2 and not game_over:
            reward2 = 10

        # if self.frame_iteration > 200:
        #     reward1 -= 0.1

        return reward1, reward2, game_over



if __name__ == '__main__':
    game = snake_game()
    
    done = False

    
    while True:
        while not done:
        # game loop
            action1 = [0, 0, 0]
            action2 = [0, 0, 0]
            action1[random.randint(0,2)] = 1
            action2[random.randint(0,2)] = 1
            reward1, reward2, done  = game.play(action1, action2)
            print(reward1, reward2)
            game.get_snake_vision(1)
        print('Score1: ', game.score1, '\nScore2: ', game.score2)
        time.sleep(2)
        game.reset()
        done = False

# action arguments: [straight, right, left]
# return arugments: game_over, reward1, reward2