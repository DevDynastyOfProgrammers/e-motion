class EntityManager:
    def __init__(self):
        self.next_entity_id = 0
        # Dict for holding all components by type
        self.components = {}

    def create_entity(self):
        """Create a new entity and return its unique ID"""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id

    def add_component(self, entity_id, component):
        """Add a component to an entity"""
        component_type = type(component)
        if component_type not in self.components:
            self.components[component_type] = {}
        self.components[component_type][entity_id] = component

    def get_component(self, entity_id, component_type):
        """Get a component of a specific type for an entity"""
        return self.components.get(component_type, {}).get(entity_id)

    def get_entities_with_component(self, component_type):
        """Get an entity and their component of a specific type"""
        if component_type in self.components:
            return self.components[component_type].items()
        return {}.items()  # Returns an empty iterator

    def get_entities_with_components(self, *component_types):
        """A generator that returns the entity ID and a tuple of its components
        for those entities that have ALL the specified component types"""
        if not component_types:
            return

        base_component_dict = min((self.components.get(ct, {}) for ct in component_types), key=len)

        for entity_id in base_component_dict:
            all_components_present = True
            result_components = []

            for ct in component_types:
                component = self.get_component(entity_id, ct)
                if component:
                    result_components.append(component)
                else:
                    all_components_present = False
                    break

            if all_components_present:
                yield entity_id, tuple(result_components)

    def remove_entity(self, entity_id):
        """Remove an entity and all its components"""
        for component_dict in self.components.values():
            if entity_id in component_dict:
                del component_dict[entity_id]
