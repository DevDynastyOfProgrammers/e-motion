import yaml
from typing import Type
from core.skill_data import (
    SkillData, ProjectileData, EffectData, AreaDamageEffectData, 
    SpawnProjectileEffectData, AnyTriggerData, AutoOnCooldownTriggerData, 
    PeriodicTriggerData, AnyEffectData
)
from core.entity_data import EntityData, EntityTransformData # <-- NEW IMPORT

class DataLoader:
    """
    Loads and parses all game data definitions from YAML files.
    """
    def __init__(self) -> None:
        self._effect_factory: dict[str, Type[EffectData]] = {
            "AREA_DAMAGE": AreaDamageEffectData,
            "SPAWN_PROJECTILE": SpawnProjectileEffectData,
        }

    # load entity info
    def load_entities(self, file_path: str) -> dict[str, EntityData]:
        """Loads entity definitions (Player, Enemy) from YAML."""
        print(f"Loading entity data from: {file_path}")
        try:
            with open(file_path, 'r') as f:
                raw_data = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"ERROR: Failed to load entities from '{file_path}': {e}")
            return {}

        entities: dict[str, EntityData] = {}
        if not isinstance(raw_data, dict):
            return {}

        for key, value in raw_data.items():
            if not isinstance(value, dict): continue
            
            try:
                # Parse Transform
                t_data = value['transform']
                transform = EntityTransformData(
                    width=float(t_data['width']),
                    height=float(t_data['height']),
                    velocity=float(t_data['velocity'])
                )
                
                # Parse Render
                color_list = value['render']['color']
                color = (int(color_list[0]), int(color_list[1]), int(color_list[2]))

                entity = EntityData(
                    id=key,
                    transform=transform,
                    max_hp=int(value['health']['max_hp']),
                    color=color,
                    tags=value.get('tags', []),
                    components=value.get('components', []),
                    skills=value.get('skills', [])
                )
                entities[key] = entity
            except (KeyError, ValueError) as e:
                print(f"WARNING: Skipping invalid entity definition '{key}': {e}")

        print(f"Successfully loaded {len(entities)} entity definitions.")
        return entities

    # load skills and projectiles info
    def load_game_data(self, file_path: str) -> tuple[dict[str, SkillData], dict[str, ProjectileData]]:
        print(f"Loading all game data from: {file_path}")
        try:
            with open(file_path, "r") as f:
                raw_data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: Data file not found at '{file_path}'")
            return {}, {}
        except yaml.YAMLError as e:
            print(f"ERROR: Failed to parse YAML file at '{file_path}': {e}")
            return {}, {}

        skills = self._load_skills(raw_data)
        projectiles = self._load_projectiles(raw_data)

        return skills, projectiles

    def _load_skills(self, raw_data: dict[str, object]) -> dict[str, SkillData]:
        """Loads all skill definitions from the raw YAML data."""
        all_skill_data: dict[str, SkillData] = {}
        if not isinstance(raw_data, dict):
            return {}

        skill_keys = [k for k in raw_data.keys() if k != "ProjectileDefinitions"]

        for skill_id in skill_keys:
            skill_info = raw_data[skill_id]
            if not isinstance(skill_info, dict):
                continue

            # Parse effects
            effects_data: list[AnyEffectData] = [] 
            if "effects" in skill_info and isinstance(skill_info["effects"], list):
                for effect_dict in skill_info["effects"]:
                    try:
                        effects_data.append(self._parse_effect(effect_dict))
                    except ValueError as e:
                        print(f"WARNING: Skipping invalid effect in skill '{skill_id}': {e}")

            # Parse trigger
            trigger_data: AnyTriggerData | None = None
            if "trigger" in skill_info and isinstance(skill_info["trigger"], dict):
                try:
                    trigger_data = self._parse_trigger(skill_info["trigger"])
                except ValueError as e:
                    print(f"WARNING: Skipping invalid trigger in skill '{skill_id}': {e}")

            # Create SkillData object
            skill_data = SkillData(
                skill_id=skill_id,
                cooldown=float(skill_info.get("cooldown", 0.0)),
                trigger=trigger_data,
                effects=effects_data,
            )
            all_skill_data[skill_id] = skill_data

        print(f"Successfully loaded {len(all_skill_data)} skills.")
        return all_skill_data

    def _load_projectiles(self, raw_data: dict[str, object]) -> dict[str, ProjectileData]:
        all_projectile_data: dict[str, ProjectileData] = {}
        if "ProjectileDefinitions" in raw_data and isinstance(raw_data["ProjectileDefinitions"], dict):
            for proj_id, proj_info in raw_data["ProjectileDefinitions"].items():
                if not isinstance(proj_info, dict): continue  
                projectile_data = ProjectileData(
                    projectile_id=proj_id,
                    components=proj_info.get("components", {}),  # type: ignore
                )
                all_projectile_data[proj_id] = projectile_data

        print(f"Successfully loaded {len(all_projectile_data)} projectiles.")
        return all_projectile_data

    def _parse_trigger(self, trigger_data: dict[str, object]) -> AnyTriggerData:
        """Parses a trigger dictionary into a TriggerData object."""
        trigger_type_str = trigger_data.get("type")
        if not trigger_type_str:
            raise ValueError("Trigger data is missing 'type' field.")

        if trigger_type_str == "AUTO_ON_COOLDOWN":
            return AutoOnCooldownTriggerData()
        elif trigger_type_str == "PERIODIC":
            interval = trigger_data.get("interval")
            if interval is None:
                raise ValueError("PERIODIC trigger missing 'interval' field.")
            return PeriodicTriggerData(interval=float(interval))  # type: ignore
        else:
            raise ValueError(f"Unknown trigger type '{trigger_type_str}'.")

    def _parse_effect(self, effect_data: dict[str, object]) -> AnyEffectData:
        """Parses a single effect dictionary into an EffectData object."""
        effect_type_str = effect_data.get("type")
        if not isinstance(effect_type_str, str):
            raise ValueError("Effect data is missing 'type' field.")

        effect_class = self._effect_factory.get(effect_type_str)
        if not effect_class:
            raise ValueError(f"Unknown effect type '{effect_type_str}'.")

        kwargs = effect_data.copy()
        del kwargs["type"]

        try:
            return effect_class(**kwargs)  # type: ignore
        except TypeError as e:
            raise ValueError(f"Mismatched data for effect type '{effect_type_str}': {e}")
