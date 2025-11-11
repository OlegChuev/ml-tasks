"""
Модуль для візуалізації результатів роботи моделей машинного навчання.

Відповідає за створення графіків для порівняння роботи різних алгоритмів
класифікації та кластеризації на двовимірних даних.
"""

import numpy as np
import matplotlib.pyplot as plt


class DataVisualizer:
    """
    Клас для візуалізації меж прийняття рішень (decision boundaries) моделей.

    Використовується для порівняння роботи моделей з учителем (Logistic Regression)
    та без учителя (KMeans) на одному графіку.
    """

    @staticmethod
    def plot_combined_boundaries(log_reg, kmeans, X_scaled, y, output_path=None):
        """
        Створює комбінований графік меж класифікації для двох моделей.

        Графік показує:
        - Кольорові зони класифікації від логістичної регресії (фон)
        - Межі кластерів від KMeans (пунктирні лінії)
        - Реальні точки даних з їх справжніми мітками

        Параметри:
            log_reg: Навчена модель логістичної регресії
            kmeans: Навчена модель KMeans
            X_scaled (ndarray): Масштабовані дані для візуалізації (N x 2)
            y (ndarray): Справжні мітки класів
            output_path (str, optional): Шлях для збереження графіка.
                                        Якщо None, графік тільки відображається.
        """
        # Крок сітки для створення mesh (чим менше, тим детальніше)
        h = .02

        # Визначення меж графіка з відступом
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

        # Створення сітки точок для візуалізації меж
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Отримання передбачень обох моделей для кожної точки сітки
        Z_lr = log_reg.predict(grid)
        Z_km = kmeans.predict(grid)

        # Перетворення передбачень назад до форми сітки
        Z_lr = Z_lr.reshape(xx.shape)
        Z_km = Z_km.reshape(xx.shape)

        # Створення фігури
        plt.figure(figsize=(10, 8))

        # Відображення меж класифікації логістичної регресії (кольорові зони)
        plt.contourf(xx, yy, Z_lr, alpha=0.3, cmap=plt.cm.Set1)

        # Відображення меж кластерів KMeans (чорні пунктирні контури)
        plt.contour(xx, yy, Z_km, linewidths=1.5, colors='k', linestyles='--', alpha=0.7)

        # Відображення реальних точок даних
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, s=30, cmap=plt.cm.tab10, edgecolor='k')

        # Підписи та заголовок
        plt.title("Порівняння меж класифікації: Logistic Regression vs KMeans", fontsize=14)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")

        # Збереження або відображення графіка
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Графік збережено у: {output_path}")
        else:
            plt.show()

        # Закриття фігури для звільнення пам'яті
        plt.close()

    @staticmethod
    def plot_clusters_scatter(X_scaled, y, kmeans_labels, output_path=None):
        """
        Створює простий scatter plot для візуалізації кластерів.

        Графік показує:
        - Точки даних, згруповані за кластерами KMeans (кольори кластерів)
        - Контури, що показують справжні класи (обведення точок)

        Параметри:
            X_scaled (ndarray): Масштабовані дані для візуалізації (N x 2)
            y (ndarray): Справжні мітки класів
            kmeans_labels (ndarray): Мітки кластерів від KMeans
            output_path (str, optional): Шлях для збереження графіка.
        """
        # Створення фігури
        plt.figure(figsize=(10, 8))

        # Визначаємо унікальні кластери
        unique_clusters = np.unique(kmeans_labels)

        # Кольорова палітра для кластерів
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        # Малюємо кожен кластер окремо
        for idx, cluster_id in enumerate(unique_clusters):
            mask = (kmeans_labels == cluster_id)
            plt.scatter(
                X_scaled[mask, 0],
                X_scaled[mask, 1],
                c=[colors[idx]],
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidths=1.5,
                label=f'Кластер {cluster_id}'
            )

        # Додаємо осі координат
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Додаємо стрілки на осях
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        # Підписи та заголовок
        plt.title("Візуалізація кластерів K-Means", fontsize=14, pad=20)
        plt.xlabel("PCA 1", fontsize=12)
        plt.ylabel("PCA 2", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)

        # Збереження або відображення графіка
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Графік кластерів збережено у: {output_path}")
        else:
            plt.show()

        # Закриття фігури для звільнення пам'яті
        plt.close()

    @staticmethod
    def plot_simple_clusters(X_scaled, y, kmeans_labels, output_path=None):
        """
        Створює спрощений scatter plot з двома основними кластерами.

        Схожий на класичну візуалізацію кластерів з чіткими групами.

        Параметри:
            X_scaled (ndarray): Масштабовані дані для візуалізації (N x 2)
            y (ndarray): Справжні мітки класів
            kmeans_labels (ndarray): Мітки кластерів від KMeans
            output_path (str, optional): Шлях для збереження графіка.
        """
        # Створення фігури з білим фоном
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')

        # Групуємо дані за кластерами (візуалізуємо тільки перші 2 кластери для спрощення)
        unique_clusters = np.unique(kmeans_labels)

        # Кольори для кластерів (червоний і синій як на скріншоті)
        cluster_colors = {
            0: '#FF6B6B',  # червоний
            1: '#4ECDC4',  # блакитний
        }

        # Малюємо точки для кожного кластера
        for cluster_id in unique_clusters[:2]:  # Візуалізуємо тільки перші 2 кластери
            mask = (kmeans_labels == cluster_id)
            color = cluster_colors.get(cluster_id, '#999999')

            # Малюємо точки (штрихування всередині кластера)
            plt.scatter(
                X_scaled[mask, 0],
                X_scaled[mask, 1],
                c=color,
                s=30,
                alpha=0.4,
                edgecolors='none',
                marker='|'
            )

            # Малюємо контур кластера
            if np.sum(mask) > 2:
                from scipy.spatial import ConvexHull
                try:
                    points = X_scaled[mask]
                    hull = ConvexHull(points)
                    # Малюємо контур
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1],
                                color=color, linewidth=2, alpha=0.8)
                    # Замикаємо контур
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    plt.plot(hull_points[:, 0], hull_points[:, 1],
                            color=color, linewidth=2.5, alpha=0.9)
                except:
                    pass

        # Налаштування осей координат
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Стрілки на осях
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        # Прибираємо мітки на осях для чистого вигляду
        ax.set_xticks([])
        ax.set_yticks([])

        # Встановлюємо межі для кращого відображення
        margin = 1.0
        ax.set_xlim(X_scaled[:, 0].min() - margin, X_scaled[:, 0].max() + margin)
        ax.set_ylim(X_scaled[:, 1].min() - margin, X_scaled[:, 1].max() + margin)

        # Збереження або відображення графіка
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Спрощений графік кластерів збережено у: {output_path}")
        else:
            plt.show()

        # Закриття фігури для звільнення пам'яті
        plt.close()