import sys
from random import randrange
from ale_python_interface import ALEInterface
from PIL import Image

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt(b'random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
    if sys.platform == 'darwin':
        import pygame
        pygame.init()
        ale.setBool('sound', False)  # Sound doesn't work on OSX
    elif sys.platform.startswith('linux'):
        ale.setBool('sound', True)
    ale.setBool('display_screen', True)

# Load the ROM file
rom_file = str.encode('data/roms/breakout.bin')
ale.loadROM(rom_file)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
print(legal_actions)

# Play 10 episodes
for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        a = legal_actions[randrange(len(legal_actions))]
        screen = ale.getScreenRGB()
        # Apply an action and get the resulting reward
        reward = ale.act(a)
        total_reward += reward
    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
    img = Image.fromarray(screen, 'RGB')
    img.show()
