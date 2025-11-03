class GameStateManager:
    """
    Управляет стеком состояний игры.
    Позволяет переключаться между различными экранами (меню, игра, пауза).
    """
    def __init__(self):
        self.states = []  # Используем список как стек состояний

    def push_state(self, state):
        """Добавляет новое состояние в вершину стека (делает его активным)."""
        self.states.append(state)

    def pop_state(self):
        """Удаляет верхнее состояние из стека."""
        if self.states:
            self.states.pop()

    def get_current_state(self):
        """Возвращает текущее активное состояние (верхнее в стеке)."""
        if self.states:
            return self.states[-1]
        return None