class EntityManager:
    def __init__(self):
        self.next_entity_id = 0
        self.components = {} # Словарь для хранения всех компонентов по типам

    def create_entity(self):
        """Создает новую сущность, возвращая ее уникальный ID."""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id

    def add_component(self, entity_id, component):
        """Добавляет компонент к сущности."""
        component_type = type(component)
        if component_type not in self.components:
            self.components[component_type] = {}
        self.components[component_type][entity_id] = component

    def get_component(self, entity_id, component_type):
        """Возвращает компонент определенного типа для сущности."""
        return self.components.get(component_type, {}).get(entity_id)

    def get_entities_with_component(self, component_type):
        """Генератор, возвращающий все сущности и их компоненты заданного типа."""
        if component_type in self.components:
            return self.components[component_type].items()
        return {}.items() # Возвращаем пустой итератор

    def get_entities_with_components(self, *component_types):
        """Генератор, возвращающий ID сущности и кортеж ее компонентов для тех сущностей,
        которые имеют ВСЕ указанные типы компонентов."""
        if not component_types:
            return

        # Находим наименьший набор сущностей для итерации (оптимизация)
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