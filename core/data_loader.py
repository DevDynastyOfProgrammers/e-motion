import yaml
from typing import Dict, Any, Type, Tuple

from core.skill_data import (
    SkillData,
    ProjectileData,
    EffectData,
    AreaDamageEffectData,
    SpawnProjectileEffectData,
    AutoOnCooldownTriggerData,
    PeriodicTriggerData,
)

class DataLoader:
    """
    Loads and parses all game data definitions from YAML files.
    """
    def __init__(self):
        self._effect_factory: Dict[str, Type[EffectData]] = {
            "AREA_DAMAGE": AreaDamageEffectData,
            "SPAWN_PROJECTILE": SpawnProjectileEffectData,
        }

    def load_game_data(self, file_path: str) -> Tuple[Dict[str, SkillData], Dict[str, ProjectileData]]:
        """
        Parses a YAML file for both skills and projectiles.

        :param file_path: The path to the data file (skills.yaml).
        :return: A tuple containing (dict_of_skills, dict_of_projectiles).
        """
        print(f"Loading all game data from: {file_path}")
        try:
            with open(file_path, 'r') as f:
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

    def _load_skills(self, raw_data: Dict[str, Any]) -> Dict[str, SkillData]:
        """Loads all skill definitions from the raw YAML data."""
        all_skill_data = {}
        skill_keys = [k for k in raw_data if k != 'ProjectileDefinitions']

        for skill_id in skill_keys:
            skill_info = raw_data[skill_id]
            
            # Parse effects
            effects_data = []
            if 'effects' in skill_info and isinstance(skill_info['effects'], list):
                for effect_dict in skill_info['effects']:
                    try:
                        effects_data.append(self._parse_effect(effect_dict))
                    except ValueError as e:
                        print(f"WARNING: Skipping invalid effect in skill '{skill_id}': {e}")
            
            # Parse trigger
            trigger_data = None
            if 'trigger' in skill_info:
                try:
                    trigger_data = self._parse_trigger(skill_info['trigger'])
                except ValueError as e:
                    print(f"WARNING: Skipping invalid trigger in skill '{skill_id}': {e}")

            # Create SkillData object
            skill_data = SkillData(
                skill_id=skill_id,
                cooldown=skill_info.get('cooldown', 0.0),
                trigger=trigger_data,
                effects=effects_data
            )
            all_skill_data[skill_id] = skill_data
            
        print(f"Successfully loaded {len(all_skill_data)} skills.")
        return all_skill_data

    def _load_projectiles(self, raw_data: Dict[str, Any]) -> Dict[str, ProjectileData]:
        """Loads all projectile definitions from the raw YAML data."""
        all_projectile_data = {}
        # Look for the specific 'ProjectileDefinitions' key
        if 'ProjectileDefinitions' in raw_data:
            for proj_id, proj_info in raw_data['ProjectileDefinitions'].items():
                projectile_data = ProjectileData(
                    projectile_id=proj_id,
                    components=proj_info.get('components', {})
                )
                all_projectile_data[proj_id] = projectile_data
        
        print(f"Successfully loaded {len(all_projectile_data)} projectiles.")
        return all_projectile_data

    def _parse_trigger(self, trigger_data: Dict[str, Any]):
        """Parses a trigger dictionary into a TriggerData object."""
        trigger_type_str = trigger_data.get('type')
        if not trigger_type_str:
            raise ValueError("Trigger data is missing 'type' field.")

        if trigger_type_str == "AUTO_ON_COOLDOWN":
            return AutoOnCooldownTriggerData()
        elif trigger_type_str == "PERIODIC":
            interval = trigger_data.get('interval')
            if interval is None:
                raise ValueError("PERIODIC trigger missing 'interval' field.")
            return PeriodicTriggerData(interval=interval)
        else:
            raise ValueError(f"Unknown trigger type '{trigger_type_str}'.")

    def _parse_effect(self, effect_data: Dict[str, Any]) -> EffectData:
        """Parses a single effect dictionary into an EffectData object."""
        effect_type_str = effect_data.get('type')
        if not effect_type_str:
            raise ValueError("Effect data is missing 'type' field.")

        effect_class = self._effect_factory.get(effect_type_str)
        if not effect_class:
            raise ValueError(f"Unknown effect type '{effect_type_str}'.")

        kwargs = effect_data.copy()
        del kwargs['type']

        try:
            return effect_class(**kwargs)
        except TypeError as e:
            raise ValueError(f"Mismatched data for effect type '{effect_type_str}': {e}")