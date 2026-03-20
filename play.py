# A modified version of the Python Pong game. Original source code: https://www.geeksforgeeks.org/python/create-a-pong-game-in-python-pygame/

# Pong

import pygame

pygame.init()

# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

# RGB values of standard colors
black = (0, 0, 0); white = (255, 255, 255); ai_colour = (0, 255, 0); cortical_colour = (255,0,0);

# Basic parameters of the screen
width, height = 150, 100
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Cortical Pong")

# parameters of the cortical system
topological_buffer = 5


clock = pygame.time.Clock()
fps = 60

def main():
    running = True

    # Defining the objects
    ai_paddle = Striker(20, 0, 2, 10, 10, ai_colour)
    cortical_paddle = Computer(width-30, 0, 2, 10, 10, cortical_colour)
    cortical_system = CorticalSystem()
    cortical_system.load_weights("models/model1.npy")
    ball = Ball(width//2, height//2, 7, 7, white)

    player_list = [ai_paddle, cortical_paddle]

    # Initial parameters of the players
    ai_paddleScore, cortical_paddleScore = 0, 0
    ai_paddleYFac, cortical_paddleYFac = 0, 0

    while running:
        screen.fill(black)

        # Collision detection
        for player in player_list:
            if pygame.Rect.colliderect(ball.getRect(), player.getRect()):
                ball.hit()

        # render ball for cortical system
        ball.display()
        
        # Updating the objects
        cortical_input = np.zeros(width + topological_buffer, heigh + topological_buffer)
        window_pixel_matrix = pygame.surfarray.array2d(screen)
        cortical_input[0:width, 0:height] = window_pixel_matrix

        for i in range(update_step):
            cortical_system.propagate(cortical_input)
        
        paddle_up_command_weight = np.sum(cortical_system.activity[::, -topological_buffer:])
        paddle_down_command_weight = np.sum(cortical_system.activity[-topological_buffer:, ::])
        
        cortical_paddleYfac = 1 if argmax([paddle_up_command_weight, paddle_down_command_weight]) == 0 else -1

        ai_paddle.update(ball.posy)
        cortical_paddle.update(cortical_paddleYFac)
        point = ball.update()

        if point == -1:
            ai_paddleScore += 1
        elif point == 1:
            cortical_paddleScore += 1

        if point:
            ball.reset()

        # render paddles
        ai_paddle.display()
        cortical_paddle.display()

        # Displaying the scores of the players
        ai_paddle.displayScore("Computer : ", ai_paddleScore, 100, 20, white)
        cortical_paddle.displayScore("Cortical : ", cortical_paddleScore, width-100, 20, white)

        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()
    pygame.quit()

