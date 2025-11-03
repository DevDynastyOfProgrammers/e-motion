import pygame
from .component import TransformComponent, RenderComponent, PlayerInputComponent

# Системы реализуют все поведение. Они работают с наборами компонентов.

class MovementSystem:
    # Этот метод пока пуст, но структура готова
    def update(self, entity_manager, delta_time):
        # Находим все сущности, у которых есть TransformComponent
        for entity, transform in entity_manager.get_entities_with_component(TransformComponent):
            # Здесь будет логика движения
            pass

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
    def update(self, entity_manager, delta_time):
        # Находим все сущности, у которых есть PlayerInputComponent и TransformComponent
        for entity, (input_comp, transform) in entity_manager.get_entities_with_components(PlayerInputComponent, TransformComponent):
            keys = pygame.key.get_pressed()
            # Используем константу скорости из settings.py, чтобы не хардкодить значения
            # Но для этого нужно будет ее передать или импортировать
            
            # Пока что для простоты можно использовать значение из компонента
            if keys[pygame.K_a]:
                transform.x -= transform.velocity * delta_time
            if keys[pygame.K_d]:
                transform.x += transform.velocity * delta_time
            if keys[pygame.K_w]:
                transform.y -= transform.velocity * delta_time
            if keys[pygame.K_s]:
                transform.y += transform.velocity * delta_time