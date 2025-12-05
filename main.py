import pygame

from core.game_state.gameplay_state import GameplayState
from core.game_state.state_manager import GameStateManager
from settings import FPS, SCREEN_HEIGHT, SCREEN_WIDTH


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # State pattern for managing game scenes (menu, gameplay, pause, game over)
        self.state_manager = GameStateManager()
        gameplay_state = GameplayState(self.state_manager)
        self.state_manager.push_state(gameplay_state)

    def run(self) -> None:
        # Game loop
        while self.running:
            delta_time = self.clock.tick(FPS) / 1000.0

            # Get the current active state
            current_state = self.state_manager.get_current_state()
            if not current_state:
                self.running = False
                continue

            # Handle events (only quit event)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False

            # Delegating event handling, updating, and rendering to the current state
            current_state.handle_events(events)
            current_state.update(delta_time)
            current_state.draw(self.screen)

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
