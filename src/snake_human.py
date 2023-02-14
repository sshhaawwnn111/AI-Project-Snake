import pygame
import time
import random

pygame.init()

class snake_game:

    def __init__(self, w=800, h=600): # initialize pygame envrionment
       

        # boundary w/h

        self.dis_width = w
        self.dis_height = h
    
        self.clock = pygame.time.Clock()

        # initial place of snake 1 and snake 2 and food // should be random

        self.snake1_x = 100
        self.snake1_y = 100

        self.snake2_x = 500
        self.snake2_y = 500
        
        self.score1 = 0
        self.score2 = 0

        # the size of a block

        self.snake_block = 10
        self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0


        # the initial direction of the snakes

        self.x1_change = 10
        self.y1_change = 0

        self.x2_change = 10
        self.y2_change = 0

        # speed of snake 

        self.snake_speed = 30
        self.snake1_list = []
        self.snake2_list = []
        self.snake1_length = 1
        self.snake2_length = 1

        self.dis=pygame.display.set_mode((self.dis_width,self.dis_height))
        pygame.display.set_caption("Snake game")

    def message(self, msg, color): # function to print message
        font_style = pygame.font.SysFont(None, 50)
        mesg = font_style.render(msg, True, color)
        self.dis.blit(mesg, [self.dis_width/2-75, self.dis_height/2-30])

    def _boundary(self):
        reward1 = 0
        reward2 = 0
        if self.snake1_x >= self.dis_width or self.snake1_x < 0 or self.snake1_y >= self.dis_height or self.snake1_y < 0:
            reward1 = -10
        
        if self.snake2_x >= self.dis_width or self.snake2_x < 0 or self.snake2_y >= self.dis_height or self.snake2_y < 0:
            reward2 = -10

        return reward1, reward2



    def _collision(self): # bump into others or suicide
        reward1 = 0
        reward2 = 0

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
    
    def play(self):
        white=(255, 255, 255)
        black=(0, 0, 0)
        blue=(0, 0, 255)
        green = (0, 255, 0)
        yellow = (255, 255, 102)

        reward1 = 0
        reward2 = 0
        game_over = False
    

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.x1_change = -self.snake_block
                    self.y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    self.x1_change = self.snake_block
                    self.y1_change = 0
                elif event.key == pygame.K_UP:
                    self.x1_change = 0
                    self.y1_change = -self.snake_block
                elif event.key == pygame.K_DOWN:
                    self.x1_change = 0
                    self.y1_change = self.snake_block
                if event.key == pygame.K_a:
                    self.x2_change = -self.snake_block
                    self.y2_change = 0
                elif event.key == pygame.K_d:
                    self.x2_change = self.snake_block
                    self.y2_change = 0
                elif event.key == pygame.K_w:
                    self.x2_change = 0
                    self.y2_change = -self.snake_block
                elif event.key == pygame.K_s:
                    self.x2_change = 0
                    self.y2_change = self.snake_block
            

        self.snake1_x += self.x1_change
        self.snake1_y += self.y1_change
        self.snake2_x += self.x2_change
        self.snake2_y += self.y2_change

        # touch the boundary
        # reward1, reward2 = self._boundary()
        if reward1 != 0 or reward2 != 0:
            game_over = True
        

        # draw background, snake movement

        self.dis.fill(white)
        pygame.draw.rect(self.dis, black, [self.foodx, self.foody, self.snake_block, self.snake_block])
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

        # draw the snake

        for x in self.snake1_list:
            #print(self.snake_length)
            pygame.draw.rect(self.dis, blue, [x[0], x[1], self.snake_block, self.snake_block])

        for x in self.snake2_list:
            #print(self.snake_length)
            pygame.draw.rect(self.dis, green, [x[0], x[1], self.snake_block, self.snake_block])


        # if snake eats the food -> generate a food position randomly

        pygame.display.update() 
        self.clock.tick(self.snake_speed)

        food = self._found_food()
        if food == 1 and not game_over:
            reward1 = 10
        elif food == 2 and not game_over:
            reward2 = 10

        return game_over, reward1, reward2



if __name__ == '__main__':
    game = snake_game()
    
    game_over = False
    while not game_over:
    # game loop
        game_over, reward1, reward2 = game.play()
        print(reward1, reward2)


    pygame.quit()
    print('reward1: ', game.score1, '\nreward2: ', game.score2)
    time.sleep(2)


# return type: game_over, reward1, reward2