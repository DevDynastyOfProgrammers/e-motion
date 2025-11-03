class BaseState:
    """
    Базовый класс для всех состояний игры (сцен).
    Определяет интерфейс, который должен реализовывать каждый дочерний класс.
    """
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def handle_events(self, events):
        """Обрабатывает события (ввод пользователя)."""
        raise NotImplementedError

    def update(self, delta_time):
        """Обновляет логику состояния."""
        raise NotImplementedError

    def draw(self, screen):
        """Отрисовывает состояние на экране."""
        raise NotImplementedError