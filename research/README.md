# Research & Development Lab

Эта папка содержит код для исследования данных, обучения моделей и генерации артефактов для проекта **e-motion**.

## Назначение
В отличие от основной папки `ml/`, которая содержит оптимизированный код для работы в реальном времени (Inference), папка `research/` используется для:
1.  **Подготовки данных** (Data Loading, Cleaning, Augmentation).
2.  **Обучения моделей** (Training Loops, Hyperparameter Tuning).
3.  **Аналитики** (Visualization, Confusion Matrices, F1-Scores).

> **Note:** Код здесь использует "тяжелые" библиотеки (`pandas`, `matplotlib`, `plotly`, `scikit-learn`, `albumentations`), которые **не включены** в финальную сборку игры для экономии ресурсов.

---

## Структура проекта

```text
research/
├── fer_loader/           # Модуль подготовки данных (Team Contribution)
│   ├── augmentations.py  # Настройки аугментаций (Albumentations)
│   ├── config.py         # Конфигурация параметров загрузки
│   ├── dataloader.py     # Логика создания DataLoader'ов и сплитов
│   ├── dataset.py        # Кастомный класс PyTorch Dataset
│   ├── logger.py         # Настройка логгирования
│   ├── main.py           # Точка входа: Распаковка -> Валидация -> Подготовка
│   └── utils.py          # Вспомогательные функции (seeds, dirs)
├── state/                # Логика "Game Director" (Адаптивная сложность)
│   ├── data/             # CSV датасеты (эмоциональные профили)
│   ├── model/            # Скрипты обучения (Pandas/Sklearn)
│   │   ├── analyzer.py   # Тяжелая аналитика и Plotly графики
│   │   └── trainer.py    # Пайплайн обучения и оценки
│   └── main.py           # Точка входа для обучения State Model
├── vision/               # Логика обучения зрения (Emotion Recognition)
│   ├── data/             # Сюда распаковывается FER-2013 (train/test)
│   ├── dataset.py        # Адаптер датасета для обучения модели
│   ├── train.py          # Цикл обучения PyTorch (EmotionCNN)
│   └── weights/          # Сохраненные веса (.pth)
└── plots/                # Сгенерированные HTML-отчеты и графики
```

---

## Руководство по запуску

Для всех команд используйте `uv run` из корня проекта.

### 1. Подготовка данных (FER-2013)
Мы используем собственный модуль `fer_loader` для распаковки, проверки структуры и применения базовых аугментаций.

**Действие:**
```bash
uv run research/fer_loader/main.py --source "path/to/downloads/FER-2013.zip"
```

*   **Что происходит:** Скрипт распаковывает архив в `research/vision/data`, создает валидационную выборку (val split) и проверяет работу `augmentations.py` и `dataloader.py`.
*   **Результат:** Готовая к обучению папка с данными.

---

### 2. Обучение модели зрения (Vision)
Обучение CNN (Convolutional Neural Network) для классификации 5 эмоций.

**Действие:**
```bash
uv run research/vision/train.py
```

*   **Вход:** Данные из `research/vision/data` (подготовленные на шаге 1).
*   **Процесс:** Обучение на GPU/CPU, применение нормализации ImageNet, валидация каждую эпоху.
*   **Выход:** Лучшие веса сохраняются в `research/vision/weights/best_emotion_model.pth`.
*   **Деплой:** Переименуйте файл в `emotion_model.pth` и поместите в корень проекта (или обновите путь в `settings.py`) для использования в игре.

---

### 3. Обучение модели состояния (State Director)
Обучение алгоритма, который переводит вектор эмоций в игровые множители (сложность).

**Действие:**
```bash
uv run research/state/main.py
```

*   **Вход:** `research/state/data/emotional_balance_dataset.csv`.
*   **Процесс:**
    1.  Анализ корреляций между эмоциями и сложностью.
    2.  Расчет центроидов (прототипов) для каждого состояния (Flow, Tension, etc.).
    3.  Генерация графиков в `research/plots/`.
*   **Выход:** Файл `synthetic_reset_prototypes.npy` сохраняется сразу в папку продакшена `ml/state/data/models/`.
*   **Деплой:** Происходит автоматически. Игра сразу подхватывает новый файл при следующем запуске.

---

## Результаты и Метрики

### Vision Model
*   **Dataset:** FER-2013 (Grayscale 48x48).
*   **Preprocessing:** ImageNet Normalization.
*   **Accuracy:** ~70% (Standard benchmark for lightweight CNN on this dataset).

### State Model
*   **Method:** Weighted Cosine Similarity + Euclidean Distance.
*   **Accuracy:** ~80% (на синтетическом датасете).
*   **Feature:** Использует **"Confidence Boost"** — уверенность Vision-модели влияет на скорость и точность смены состояния игры.

