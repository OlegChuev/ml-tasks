"""
Модуль для завантаження та підготовки даних про цифри (digits dataset).

Цей модуль відповідає за:
- Завантаження вбудованого датасету цифр з scikit-learn
- Зменшення розмірності даних за допомогою PCA
- Нормалізацію даних
- Розділення на тренувальну та тестову вибірки
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DigitsDataLoader:
    """
    Клас для завантаження та попередньої обробки датасету рукописних цифр.

    Виконує повний цикл підготовки даних:
    1. Завантаження датасету (8x8 зображення цифр 0-9)
    2. Зменшення розмірності з 64 ознак до 2 (для візуалізації)
    3. Стандартизація даних
    4. Розділення на train/test множини
    """

    def __init__(self, n_components=2, test_size=0.3, random_state=42):
        """
        Ініціалізація завантажувача даних.

        Параметри:
            n_components (int): Кількість головних компонент для PCA (за замовчуванням 2 для візуалізації)
            test_size (float): Частка даних для тестової вибірки (0.3 = 30%)
            random_state (int): Seed для відтворюваності результатів
        """
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        """
        Завантажує та обробляє датасет рукописних цифр.

        Процес обробки:
        1. Завантажує датасет цифр (1797 зразків, 64 ознаки кожен)
        2. Застосовує PCA для зменшення розмірності до n_components
        3. Стандартизує дані (mean=0, std=1)
        4. Розділяє на тренувальну та тестову вибірки

        Повертає:
            tuple: (X_train, X_test, y_train, y_test, X_scaled, y)
                - X_train: тренувальні ознаки
                - X_test: тестові ознаки
                - y_train: тренувальні мітки
                - y_test: тестові мітки
                - X_scaled: всі масштабовані дані (для візуалізації)
                - y: всі мітки класів
        """
        # Завантаження вбудованого датасету цифр
        digits = load_digits()
        X, y = digits.data, digits.target

        # Зменшення розмірності за допомогою PCA (з 64 до 2 ознак)
        pca = PCA(n_components=self.n_components)
        X_reduced = pca.fit_transform(X)

        # Стандартизація даних (нормалізація до mean=0, std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)

        # Розділення на тренувальну (70%) та тестову (30%) вибірки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test, X_scaled, y