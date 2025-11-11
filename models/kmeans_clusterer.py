"""
Модуль з реалізацією кластеризатора на основі алгоритму K-Means.

K-Means - це алгоритм машинного навчання без учителя (unsupervised learning),
який групує дані у кластери на основі їх схожості без використання міток класів.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np


class KMeansClusterer:
    """
    Клас-обгортка для моделі кластеризації K-Means.

    Використовує алгоритм K-Means для групування даних у 10 кластерів (за кількістю цифр).
    Після кластеризації, кожному кластеру призначається мітка на основі найчастішої цифри
    в цьому кластері (для оцінки точності).
    """

    def __init__(self, n_clusters=10, random_state=42):
        """
        Ініціалізує модель K-Means.

        Параметри:
            n_clusters (int): Кількість кластерів (10 для цифр 0-9)
            random_state (int): Seed для відтворюваності результатів
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def train(self, X):
        """
        Навчає модель кластеризації (без використання міток).

        K-Means групує дані лише за їх координатами, не знаючи про справжні класи.

        Параметри:
            X (ndarray): Дані для кластеризації (features)
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Передбачає номер кластера для нових даних.

        Параметри:
            X (ndarray): Дані для класифікації

        Повертає:
            ndarray: Номери кластерів (від 0 до n_clusters-1)
        """
        return self.model.predict(X)

    def accuracy(self, X_train, y_train):
        """
        Обчислює приблизну точність кластеризації.

        Оскільки K-Means не знає про справжні мітки, ми призначаємо кожному кластеру
        мітку класу, який найчастіше зустрічається в цьому кластері (mode).
        Це дозволяє оцінити, наскільки добре кластеризація відповідає реальним класам.

        Параметри:
            X_train (ndarray): Тренувальні дані
            y_train (ndarray): Справжні мітки класів

        Повертає:
            float: Приблизна точність (значення від 0 до 1)
        """
        # Створюємо масив для зберігання призначених міток
        labels = np.zeros_like(self.model.labels_)

        # Для кожного кластера знаходимо найпопулярнішу цифру
        for i in range(10):
            mask = (self.model.labels_ == i)
            if np.any(mask):
                # Призначаємо кластеру мітку, яка найчастіше зустрічається в ньому
                labels[mask] = mode(y_train[mask], keepdims=False).mode

        return accuracy_score(y_train, labels)
