import math
import pygame
import random
from .component import *
from settings import SCREEN_WIDTH, SCREEN_HEIGHT


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


class SpellAuraSystem:
    """Processes spell auras that damage nearby entities by targeting the tag"""
    def update(self, entity_manager, delta_time):
        aura_entities = entity_manager.get_entities_with_components(SpellAuraComponent, TransformComponent)
        
        target_entities = list(entity_manager.get_entities_with_components(HealthComponent, TransformComponent, TagComponent))

        for aura_entity, (aura, aura_transform) in aura_entities:
            aura.time_since_last_tick += delta_time

            if aura.time_since_last_tick < aura.tick_rate:
                continue
            
            aura.time_since_last_tick = 0.0

            for target_entity, (health, target_transform, tag) in target_entities:
                # Skip non-target tags
                if tag.tag != aura.target_tag:
                    continue

                # Distance from aura center to target
                distance = math.hypot(
                    aura_transform.x - target_transform.x,
                    aura_transform.y - target_transform.y
                )

                if distance <= aura.radius:
                    health.current_hp -= aura.damage
                    print(f"Entity {target_entity} took {aura.damage} damage! Current HP: {health.current_hp}")


class DeathSystem:
    def update(self, entity_manager):
        entities_to_remove = []
        
        for entity, health in entity_manager.get_entities_with_component(HealthComponent):
            if health.current_hp <= 0:
                entities_to_remove.append(entity)
        
        for entity in entities_to_remove:
            entity_manager.remove_entity(entity)
            print(f"Entity {entity} has died and been removed from the game.")


class EnemySpawningSystem:
    """
    Manages the spawning of enemy waves over time.
    Uses an EntityFactory to create enemies.
    """
    def __init__(self, entity_factory):
        self.factory = entity_factory
        
        # single enemy spawn parameters
        self.time_since_last_single_spawn = 0.0
        self.single_spawn_interval = 3.0 # Spawn a new enemy every 3 seconds

        # enemy group spawn parameters
        self.time_since_last_group_spawn = 0.0
        self.group_spawn_interval = 10.0
        self.group_size = 5
        self.group_spawn_radius = 100.0

    def update(self, delta_time):
        # single enemy spawn processing
        self.time_since_last_single_spawn += delta_time
        if self.time_since_last_single_spawn >= self.single_spawn_interval:
            self.time_since_last_single_spawn = 0.0
            self._spawn_single_enemy()

        # enemy group spawn processing
        self.time_since_last_group_spawn += delta_time
        if self.time_since_last_group_spawn >= self.group_spawn_interval:
            self.time_since_last_group_spawn = 0.0
            self._spawn_enemy_group()

    def _get_random_offscreen_position(self):
        side = random.randint(0, 3)
        if side == 0: # Top
            x, y = random.randint(0, SCREEN_WIDTH), -50
        elif side == 1: # Right
            x, y = SCREEN_WIDTH + 50, random.randint(0, SCREEN_HEIGHT)
        elif side == 2: # Bottom
            x, y = random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + 50
        else: # Left
            x, y = -50, random.randint(0, SCREEN_HEIGHT)
        return x, y

    def _spawn_single_enemy(self):
        # Choose a random position outside the screen borders
        x, y = self._get_random_offscreen_position()
        
        print(f"Spawning enemy at ({x}, {y})")
        self.factory.create_enemy(x, y)

    def _spawn_enemy_group(self):
        """Spawns a group of enemies in a cluster at a random off-screen location."""
        # Choose a random central position for a group outside the screen borders
        center_x, center_y = self._get_random_offscreen_position()
        print(f"Spawning GROUP of {self.group_size} enemies around ({center_x}, {center_y})")

        # Spawning enemies in area around the center point
        for _ in range(self.group_size):
            offset_x = random.uniform(-self.group_spawn_radius, self.group_spawn_radius)
            offset_y = random.uniform(-self.group_spawn_radius, self.group_spawn_radius)
            
            self.factory.create_enemy(center_x + offset_x, center_y + offset_y)