import pygame
from settings import *
from core.game_state.state_manager import GameStateManager
from core.game_state.gameplay_state import GameplayState

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Паттерн "Состояние" (State) для управления сценами (меню, игра, геймовер)
        self.state_manager = GameStateManager()
        gameplay_state = GameplayState(self.state_manager)
        self.state_manager.push_state(gameplay_state)

    def run(self):
        # Классический "Игровой цикл"
        while self.running:
            delta_time = self.clock.tick(FPS) / 1000.0
            
            # Получаем текущее активное состояние (например, GameplayState)
            current_state = self.state_manager.get_current_state()
            if not current_state:
                self.running = False
                continue

            # 1. Обработка событий
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Делегируем обработку событий, обновление и отрисовку текущему состоянию
            current_state.handle_events(events)
            current_state.update(delta_time)
            current_state.draw(self.screen)

            pygame.display.flip()
        
        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()