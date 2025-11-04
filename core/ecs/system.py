import math
import pygame
from .component import TransformComponent, RenderComponent, PlayerInputComponent, AIComponent


class EnemyChaseSystem:
    """Processes AI-controlled enemies to chase the player"""
    def update(self, entity_manager, player_transform, delta_time):

        if not player_transform:
            return
        
        # Find all enemies with AIComponent and TransformComponent
        for entity, (ai_comp, transform) in entity_manager.get_entities_with_components(AIComponent, TransformComponent):
            dx = player_transform.x - transform.x
            dy = player_transform.y - transform.y

            alpha = math.atan2(dy, dx)

            transform.x += math.cos(alpha) * transform.velocity * delta_time
            transform.y += math.sin(alpha) * transform.velocity * delta_time


class RenderSystem:
    """Renders all entities with TransformComponent and RenderComponent to the screen"""
    def update(self, entity_manager, screen):
        # Clear the screen
        screen.fill((20, 20, 20))
        
        # Find all entities with TransformComponent and RenderComponent
        entities_to_render = entity_manager.get_entities_with_components(TransformComponent, RenderComponent)
        
        for entity, (transform, render) in entities_to_render:
            rect = pygame.Rect(transform.x, transform.y, transform.width, transform.height)
            pygame.draw.rect(screen, render.color, rect)

class PlayerInputSystem:
    """Processes player input and moves the player entity accordingly"""
    def update(self, entity_manager, delta_time):
        # Find all players with PlayerInputComponent and TransformComponent (there will be only 1)
        for entity, (input_comp, transform) in entity_manager.get_entities_with_components(PlayerInputComponent, TransformComponent):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                transform.x -= transform.velocity * delta_time
            if keys[pygame.K_d]:
                transform.x += transform.velocity * delta_time
            if keys[pygame.K_w]:
                transform.y -= transform.velocity * delta_time
            if keys[pygame.K_s]:
                transform.y += transform.velocity * delta_time