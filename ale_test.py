from ale_python_interface.ale_python_interface import ALEInterface
import numpy as np
# import pygame

ale = ALEInterface()

ale.setInt(b"random_seed", 123)
ale.setBool(b'display_screen', True)
ale.loadROM(str.encode('data/roms/breakout.bin'))

random_seed = ale.getInt(b"random_seed")
print("random_seed: " + str(random_seed))

legal_actions = ale.getMinimalActionSet()

(screen_width, screen_height) = ale.getScreenDims()
print("width/height: " + str(screen_width) + "/" + str(screen_height))

#  init pygame
# pygame.init()
# print(ale.getScreenGrayscale().flatten().shape)
# screen = pygame.display.set_mode((160, 210))
# pygame.display.set_caption("Arcade Learning Environment Random Agent Display")

# pygame.display.flip()

episode = 0
total_reward = 0.0
while(episode < 10):
    a = legal_actions[np.random.randint(legal_actions.size)]
    reward = ale.act(a)
    total_reward += reward

    # numpy_surface = np.frombuffer(screen.get_buffer(), dtype=np.int32)
    # rgb = ale.getScreenRGB()
    # print(rgb.shape)
    # print(rgb.mean())
    # # print(tuple(map(lambda s: s / 3, rgb.shape)))
    # # print(np.frombuffer(screen.get_buffer()).shape)
    # print(pygame.PixelArray(screen).shape)
    # bv = screen.get_buffer()
    # bv.write(rgb.tostring(), 0)
    # pygame.display.flip()
    if(ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) +
              " Episode Frame Number: " + str(episode_frame_number))
        print("Episode " + str(episode) +
              " ended with score: " + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1
