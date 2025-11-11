"""
Модуль з реалізацією класифікатора на основі логістичної регресії.

Логістична регресія - це алгоритм машинного навчання з учителем (supervised learning),
який використовується для задач класифікації. Підходить для багатокласової класифікації.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogisticRegressionClassifier:
    """
    Клас-обгортка для моделі логістичної регресії.

    Використовує алгоритм логістичної регресії з scikit-learn для класифікації
    рукописних цифр. Модель навчається на основі міток класів (supervised learning).
    """

    def __init__(self):
        """
        Ініціалізує модель логістичної регресії.

        Параметри:
            max_iter (int): Максимальна кількість ітерацій для збіжності алгоритму.
                           Встановлено 1000 для гарантованої збіжності на складних даних.
        """
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        """
        Навчає модель на тренувальних даних.

        Параметри:
            X_train (ndarray): Тренувальні ознаки (features)
            y_train (ndarray): Тренувальні мітки класів (labels)
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Робить передбачення класів для нових даних.

        Параметри:
            X (ndarray): Дані для класифікації

        Повертає:
            ndarray: Передбачені мітки класів
        """
        return self.model.predict(X)

    def accuracy(self, X_test, y_test):
        """
        Обчислює точність моделі на тестових даних.

        Точність (accuracy) - це відношення правильно класифікованих зразків
        до загальної кількості зразків.

        Параметри:
            X_test (ndarray): Тестові ознаки
            y_test (ndarray): Справжні мітки для тестових даних

        Повертає:
            float: Точність класифікації (значення від 0 до 1)
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

