import pygame
import numpy as np
import src.pong

pygame.init()

# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

WHITE = (255, 255, 255)

# Basic parameters of the screen
width, height = 150, 100
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong")


clock = pygame.time.Clock()    
fps = 5
nFrames = 300 # arbitrary number

buffer_topology = 5 ## amount of blank canvas for buttons to wrap around
data_matrix = np.zeros([width + buffer_topology,height + buffer_topology,nFrames])

def main():
    running = True

    # Defining the objects
    computer_0 = Computer(20, 0, 2, 10, 10, WHITE)
    computer_1 = Computer(width-30, 0, 2, 10, 10, WHITE)
    ball = Ball(width//2, height//2, 3, 7, WHITE)

    list_of_computers = [computer_0, computer_1]
    i = 0
    while running:
        screen.fill([0,0,0])

        # Collision detection
        for ai in list_of_computers:
            if pygame.Rect.colliderect(ball.getRect(), ai.getRect()):
                ball.hit()

        # get the position before the frame 
        computer_pre = computer_0.posy
        # Updating the objects
        computer_0.update(ball.posy)
        computer_1.update(ball.posy)
        point = ball.update()

        if point:   
            ball.reset()
        
        # get the position after the frame
        computer_post = computer_0.posy
        
        # Displaying the objects on the screen
        ball.display()
        pygame.display.update()
        clock.tick(fps)
        
        # collect screen data
        
        window_pixel_matrix = pygame.surfarray.array2d(screen)
        
        # augment paddle data to topological neural field
        button = np.sign(computer_post - computer_pre)
        data_matrix[0:width, 0:height, i] = window_pixel_matrix
        if button == 1:
            data_matrix[-buffer_topology,:,i] = 1
            data_matrix[:,-buffer_topology,i] = -1
        elif button == 0:
            data_matrix[-buffer_topology,:,i] = -1
            data_matrix[:,-buffer_topology,i] = 1
        
        i += 1
        if i > np.shape(data_matrix)[-1]:
            raise ValueError("Training data saturated")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        np.save("data/data", data_matrix)
    pygame.quit()

