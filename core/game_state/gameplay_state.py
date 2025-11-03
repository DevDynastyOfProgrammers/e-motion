from .base_state import BaseState
from core.ecs.entity import EntityManager
from core.ecs.component import *
from core.ecs.system import *

class GameplayState(BaseState):
    def __init__(self, state_manager):
        super().__init__(state_manager)
        self.entity_manager = EntityManager()
        
        # Инициализируем системы
        self.render_system = RenderSystem()
        self.player_input_system = PlayerInputSystem()
        # ... другие системы

        # Создаем сущности (Паттерн "Фабрика" можно применить здесь в будущем)
        self.create_player(100, 100)
        self.create_enemy(400, 300)

    def create_player(self, x, y):
        player = self.entity_manager.create_entity()
        self.entity_manager.add_component(player, TransformComponent(x, y, 30, 30, velocity=200))
        self.entity_manager.add_component(player, RenderComponent(color=(0, 150, 255)))
        self.entity_manager.add_component(player, PlayerInputComponent())
        self.entity_manager.add_component(player, HealthComponent(100, 100))

    def create_enemy(self, x, y):
        enemy = self.entity_manager.create_entity()
        self.entity_manager.add_component(enemy, TransformComponent(x, y, 25, 25, velocity=100))
        self.entity_manager.add_component(enemy, RenderComponent(color=(255, 50, 50)))
        self.entity_manager.add_component(enemy, AIComponent())
        self.entity_manager.add_component(enemy, HealthComponent(50, 50))

    def handle_events(self, events):
        # Этот метод больше не нужен для системы ввода, так как pygame.key.get_pressed()
        # работает без цикла событий. Оставляем его для будущих нужд (например, клики мыши).
        pass

    def update(self, delta_time):
        # Обновляем все системы в определенном порядке
        self.player_input_system.update(self.entity_manager, delta_time)
        # self.movement_system.update(self.entity_manager, delta_time)
        # self.ai_system.update(self.entity_manager, delta_time)
        
    def draw(self, screen):
        self.render_system.update(self.entity_manager, screen)