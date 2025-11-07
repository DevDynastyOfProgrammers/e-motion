
import math
import pygame
import random
from .component import TransformComponent, RenderComponent, PlayerInputComponent, AIComponent, \
    HealthComponent, TagComponent, SkillSetComponent, ProjectileComponent, DamageOnCollisionComponent, LifetimeComponent
from core.events import PlayerMoveIntentEvent, EntityDeathEvent, ApplyAreaDamageEvent, \
    ApplyDirectDamageEvent, RequestEntityRemovalEvent, SpawnProjectileEvent, RequestSkillActivationEvent
from core.skill_data import AreaDamageEffectData, SpawnProjectileEffectData, AutoOnCooldownTriggerData, \
    PeriodicTriggerData
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

            dist = math.hypot(dx, dy)
            if dist > 0:
                dx, dy = dx / dist, dy / dist

            transform.x += dx * transform.velocity * delta_time
            transform.y += dy * transform.velocity * delta_time


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
    """Processes player input and posts movement intent events."""
    def __init__(self, event_manager):
        self.event_manager = event_manager

    def update(self, entity_manager):
        # Find the player entity
        for entity, (input_comp, transform) in entity_manager.get_entities_with_components(PlayerInputComponent, TransformComponent):
            keys = pygame.key.get_pressed()
            dx, dy = 0, 0
            if keys[pygame.K_a]: dx -= 1
            if keys[pygame.K_d]: dx += 1
            if keys[pygame.K_w]: dy -= 1
            if keys[pygame.K_s]: dy += 1
            if dx != 0 and dy != 0:
                length = math.sqrt(dx**2 + dy**2)
                dx /= length
                dy /= length
            if dx != 0 or dy != 0:
                self.event_manager.post(PlayerMoveIntentEvent(entity, (dx, dy)))


class MovementSystem:
    """
    Handles movement requests and updates TransformComponents.
    This system will be expanded to handle all physics-based movement.
    """
    def __init__(self, event_manager, entity_manager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        
        # Subscribe to movement events
        self.event_manager.subscribe(PlayerMoveIntentEvent, self.on_player_move)

        # A dictionary to store movement requests for the current frame
        self.movement_requests = {}

    def on_player_move(self, event):
        """
        Callback that receives PlayerMoveIntentEvent.
        Stores the movement direction for processing in the update phase.
        """
        self.movement_requests[event.entity_id] = event.direction

    def update(self, delta_time):
        """
        Apply the stored movement requests to the entities' TransformComponents.
        """
        for entity_id, direction in self.movement_requests.items():
            transform = self.entity_manager.get_component(entity_id, TransformComponent)
            if transform:
                transform.x += direction[0] * transform.velocity * delta_time
                transform.y += direction[1] * transform.velocity * delta_time
        
        # Clear requests for the next frame
        self.movement_requests.clear()

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

class SkillSystem:
    """
    Manages skill state (cooldowns, timers) and checks trigger
    conditions to automatically request skill activations.
    """
    def __init__(self, event_manager, entity_manager, skill_definitions):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.skill_definitions = skill_definitions

    def update(self, delta_time):
        # Iterate through all entities that can use skills
        for entity, (skill_set,) in self.entity_manager.get_entities_with_components(SkillSetComponent):
            
            # 1. Update all cooldowns and timers
            for skill_id in skill_set.skills:
                if skill_set.cooldowns[skill_id] > 0:
                    skill_set.cooldowns[skill_id] -= delta_time
                skill_set.periodic_timers[skill_id] += delta_time

            # 2. Check all trigger conditions
            for skill_id in skill_set.skills:
                skill_data = self.skill_definitions.get(skill_id)
                if not skill_data or not skill_data.trigger:
                    continue

                # Check if skill is ready (off cooldown)
                if skill_set.cooldowns[skill_id] > 0:
                    continue
                
                trigger_data = skill_data.trigger
                should_activate = False

                # Logic for different trigger types
                if isinstance(trigger_data, AutoOnCooldownTriggerData):
                    should_activate = True
                
                elif isinstance(trigger_data, PeriodicTriggerData):
                    if skill_set.periodic_timers[skill_id] >= trigger_data.interval:
                        should_activate = True

                # If any trigger condition is met, request activation
                if should_activate:
                    self.event_manager.post(RequestSkillActivationEvent(entity, skill_id))
                    # Reset periodic timer if it was a periodic activation
                    if isinstance(trigger_data, PeriodicTriggerData):
                        skill_set.periodic_timers[skill_id] = 0.0


class SkillExecutionSystem:
    """Listens for skill activation requests and executes their effects."""
    def __init__(self, event_manager, entity_manager, skill_definitions):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.skill_definitions = skill_definitions
        self.event_manager.subscribe(RequestSkillActivationEvent, self.on_skill_request)
    
    def on_skill_request(self, event):
        skill_set = self.entity_manager.get_component(event.entity_id, SkillSetComponent)
        transform = self.entity_manager.get_component(event.entity_id, TransformComponent)
        skill_data = self.skill_definitions.get(event.skill_id)

        if not all([skill_set, transform, skill_data]):
            return
        
        # Check cooldown
        if skill_set.cooldowns.get(event.skill_id, 0) > 0:
            return # Skill is on cooldown
        
        # Set cooldown
        skill_set.cooldowns[event.skill_id] = skill_data.cooldown

        # Execute effects
        print(f"Executing skill '{skill_data.skill_id}' for entity {event.entity_id}")
        for effect in skill_data.effects:
            if isinstance(effect, AreaDamageEffectData):
                self.event_manager.post(ApplyAreaDamageEvent(
                    caster_id=event.entity_id,
                    caster_pos=(transform.x, transform.y),
                    effect_data=effect
                ))
            elif isinstance(effect, SpawnProjectileEffectData):
                self.event_manager.post(SpawnProjectileEvent(
                    caster_id=event.entity_id,
                    effect_data=effect
                ))


class DamageSystem:
    """Listens for damage events and applies them."""
    def __init__(self, event_manager, entity_manager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.event_manager.subscribe(ApplyAreaDamageEvent, self.on_area_damage)
        self.event_manager.subscribe(ApplyDirectDamageEvent, self.on_direct_damage)

    def on_area_damage(self, event):
        """Applies damage to all valid targets in a radius."""
        effect_data = event.effect_data
        caster_pos = event.caster_pos
        target_entities = self.entity_manager.get_entities_with_components(HealthComponent, TransformComponent, TagComponent)

        for target_id, (health, transform, tag) in target_entities:
            if target_id == event.caster_id: continue
            if tag.tag == effect_data.target_tag:
                distance = math.hypot(caster_pos[0] - transform.x, caster_pos[1] - transform.y)
                if distance <= effect_data.radius:
                    health.current_hp -= effect_data.damage
                    print(f"Entity {target_id} took {effect_data.damage} AREA damage! HP: {health.current_hp}")
    
    def on_direct_damage(self, event):
        """Applies damage to a single specific target."""
        health = self.entity_manager.get_component(event.target_id, HealthComponent)
        if health:
            health.current_hp -= event.damage
            print(f"Entity {event.target_id} took {event.damage} DIRECT damage! HP: {health.current_hp}")


class DeathSystem:
    """Manages entity death and removal based on health or requests."""
    def __init__(self, event_manager, entity_manager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.event_manager.subscribe(RequestEntityRemovalEvent, self.on_removal_request)
    
    def on_removal_request(self, event):
        """Handles explicit requests to remove an entity."""
        if self.entity_manager.remove_entity(event.entity_id):
            # print(f"Entity {event.entity_id} was removed by request.")
            pass

    def update(self):
        """Checks for entities with zero or less health."""
        entities_to_remove = []
        for entity, health in self.entity_manager.get_entities_with_component(HealthComponent):
            if health.current_hp <= 0:
                entities_to_remove.append(entity)
        
        for entity in entities_to_remove:
            if self.entity_manager.remove_entity(entity):
                self.event_manager.post(EntityDeathEvent(entity))
                print(f"Entity {entity} has DIED and been removed from the game.")

class ProjectileSpawningSystem:
    """Listens for SpawnProjectileEvent and creates projectiles."""
    def __init__(self, event_manager, entity_manager, projectile_definitions, factory):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        self.projectile_definitions = projectile_definitions
        self.factory = factory
        self.event_manager.subscribe(SpawnProjectileEvent, self.on_spawn_projectile)
    
    def on_spawn_projectile(self, event):
        caster_transform = self.entity_manager.get_component(event.caster_id, TransformComponent)
        if not caster_transform: return
        
        projectile_data = self.projectile_definitions.get(event.effect_data.projectile_id)
        if not projectile_data:
            print(f"ERROR: Unknown projectile_id '{event.effect_data.projectile_id}'")
            return
        
        # --- Targeting Logic ---
        direction = (1, 0) # Default direction (right)
        if event.effect_data.target_logic == "nearest_enemy":
            target = self._find_nearest_enemy(caster_transform.x, caster_transform.y)
            if target:
                target_transform = self.entity_manager.get_component(target, TransformComponent)
                dx = target_transform.x - caster_transform.x
                dy = target_transform.y - caster_transform.y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    direction = (dx / dist, dy / dist)
        
        self.factory.create_projectile(caster_transform.x, caster_transform.y, direction, projectile_data)

    def _find_nearest_enemy(self, x, y):
        """Finds the entity with an AIComponent closest to the given point."""
        # This is O(N) and not optimal, but fine for now.
        enemies = self.entity_manager.get_entities_with_components(AIComponent, TransformComponent)
        closest_enemy = None
        min_dist_sq = float('inf')

        for enemy_id, (ai, transform) in enemies:
            dist_sq = (transform.x - x)**2 + (transform.y - y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_enemy = enemy_id
        return closest_enemy


class ProjectileMovementSystem:
    """Moves all entities with a ProjectileComponent."""
    def update(self, entity_manager, delta_time):
        for entity, (proj, transform) in entity_manager.get_entities_with_components(ProjectileComponent, TransformComponent):
            transform.x += proj.dx * transform.velocity * delta_time
            transform.y += proj.dy * transform.velocity * delta_time

class ProjectileImpactSystem:
    """Checks for projectile collisions and posts damage events."""
    def __init__(self, event_manager, entity_manager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager

    def update(self):
        projectiles = list(self.entity_manager.get_entities_with_components(ProjectileComponent, TransformComponent, DamageOnCollisionComponent))
        targets = list(self.entity_manager.get_entities_with_components(HealthComponent, TransformComponent, TagComponent))

        for proj_id, (proj, proj_trans, proj_damage) in projectiles:
            for target_id, (health, target_trans, tag) in targets:
                if tag.tag != proj_damage.target_tag:
                    continue

                # Simple radius-based collision check
                if math.hypot(proj_trans.x - target_trans.x, proj_trans.y - target_trans.y) < target_trans.width:
                    self.event_manager.post(ApplyDirectDamageEvent(target_id, proj_damage.damage))
                    # Destroy projectile on impact
                    self.event_manager.post(RequestEntityRemovalEvent(proj_id))
                    break # Projectile can only hit one target

class LifetimeSystem:
    """Decrements lifetime on components and removes entities when it expires."""
    def __init__(self, event_manager, entity_manager):
        self.event_manager = event_manager
        self.entity_manager = entity_manager
        
    def update(self, delta_time):
        for entity, (lifetime,) in self.entity_manager.get_entities_with_components(LifetimeComponent):
            lifetime.time_remaining -= delta_time
            if lifetime.time_remaining <= 0:
                self.event_manager.post(RequestEntityRemovalEvent(entity))