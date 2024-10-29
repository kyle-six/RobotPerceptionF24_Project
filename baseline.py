# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
from sklearn.neighbors import BallTree

from mazeGraph import MazeGraph, Node

# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()

        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to

        # Builds Graph, if it doesn't exist
        self.mazeGraph = MazeGraph()

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    #self.show_target_images()
                    pass
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h / 2), 0), (int(h / 2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w / 2)), (h, int(w / 2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(
            concat_img,
            "Front View",
            (h_offset, w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Right View",
            (int(h / 2) + h_offset, w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Back View",
            (h_offset, int(w / 2) + w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Left View",
            (int(h / 2) + h_offset, int(w / 2) + w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )

        cv2.imshow(f"KeyboardPlayer:target_images", concat_img)
        cv2.waitKey(1)

    def find_shortest_path_AND_create_video(self):
        targets = self.get_target_images()
        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return
        
        ######## Calculate shortest path and create video from target image
        self.mazeGraph.init_navigation(self.get_target_images()[0])

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()
        self.find_shortest_path_AND_create_video()
    
    def show_video(self):
        cap = cv2.VideoCapture(self.mazeGraph.path_video_path)
        while(cap.isOpened()):
        # Capture each frame
            # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                # Display the resulting frame
                    cv2.imshow('Frame', frame)
                    
                # Press Q on keyboard to exit
                    if cv2.waitKey(55) & 0xFF == ord('q'):
                        break
            # Break the loop
                else:
                    break
        cap.release()
                    

    def see(self, fpv):
        """
        Set the first-person view input
        """
        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[
                1::-1
            ]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, "RGB")

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                pass
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.show_target_images()
                
        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game

    # Start the game with the KeyboardPlayerPyGame player
    player = KeyboardPlayerPyGame()
    vis_nav_game.play(the_player=player)

