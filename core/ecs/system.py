import math
import pygame
from .component import TransformComponent, RenderComponent, PlayerInputComponent, AIComponent

# Системы реализуют все поведение. Они работают с наборами компонентов.

class EnemyChaseSystem:
    """Система, которая заставляет врагов с AIComponent преследовать игрока."""
    def update(self, entity_manager, player_transform, delta_time):

        if not player_transform:
            return
        
        # Находим всех врагов с AIComponent и TransformComponent
        for entity, (ai_comp, transform) in entity_manager.get_entities_with_components(AIComponent, TransformComponent):
            dx = player_transform.x - transform.x
            dy = player_transform.y - transform.y

            alpha = math.atan2(dy, dx)

            transform.x += math.cos(alpha) * transform.velocity * delta_time
            transform.y += math.sin(alpha) * transform.velocity * delta_time


class RenderSystem:
    def update(self, entity_manager, screen):
        # Очистка экрана
        screen.fill((20, 20, 20))
        
        # Находим все сущности, у которых есть TransformComponent И RenderComponent
        entities_to_render = entity_manager.get_entities_with_components(TransformComponent, RenderComponent)
        
        for entity, (transform, render) in entities_to_render:
            rect = pygame.Rect(transform.x, transform.y, transform.width, transform.height)
            pygame.draw.rect(screen, render.color, rect)

class PlayerInputSystem:
    """Система для обработки ввода игрока и перемещения игрока."""
    def update(self, entity_manager, delta_time):
        # Находим всех игроков с PlayerInputComponent и TransformComponent (будет только 1)
        for entity, (input_comp, transform) in entity_manager.get_entities_with_components(PlayerInputComponent, TransformComponent):
            keys = pygame.key.get_pressed()            
            # Пока что для простоты можно использовать значение из компонента
            if keys[pygame.K_a]:
                transform.x -= transform.velocity * delta_time
            if keys[pygame.K_d]:
                transform.x += transform.velocity * delta_time
            if keys[pygame.K_w]:
                transform.y -= transform.velocity * delta_time
            if keys[pygame.K_s]:
                transform.y += transform.velocity * delta_time