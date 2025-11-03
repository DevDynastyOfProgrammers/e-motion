# Базовый класс для всех компонентов. По сути, просто маркер.
class Component:
    pass

# Компоненты - это просто контейнеры данных (POD - Plain Old Data)

class TransformComponent(Component):
    def __init__(self, x, y, width, height, velocity=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = velocity

class RenderComponent(Component):
    def __init__(self, color, layer=0):
        self.color = color # Для PoC просто цвет, потом будет спрайт
        self.layer = layer

class PlayerInputComponent(Component):
    # Маркерный компонент, который говорит системе ввода, что эту сущность надо слушать
    pass

class AIComponent(Component):
    # Маркерный компонент, который говорит системе ИИ, что эту сущность надо обрабатывать
    def __init__(self, ai_type="chase_player"):
        self.ai_type = ai_type

class HealthComponent(Component):
    def __init__(self, current_hp, max_hp):
        self.current_hp = current_hp
        self.max_hp = max_hp