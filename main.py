"""
Головний модуль для порівняння алгоритмів класифікації та кластеризації цифр.

Цей скрипт демонструє різницю між навчанням з учителем (Logistic Regression)
та навчанням без учителя (K-Means) на датасеті рукописних цифр.

Результати роботи:
- Виводить точність обох моделей у консоль
- Зберігає візуалізацію меж класифікації у директорію output/
- Зберігає результати метрик у текстовий файл
"""

import os
from datetime import datetime
from data_loader import DigitsDataLoader
from models.logistic_regression_classifier import LogisticRegressionClassifier
from models.kmeans_clusterer import KMeansClusterer
from visualizer import DataVisualizer


def main():
    """
    Головна функція для виконання порівняльного аналізу моделей.

    Виконує наступні кроки:
    1. Завантаження та підготовка даних
    2. Навчання моделі логістичної регресії
    3. Навчання моделі K-Means
    4. Обчислення та виведення точності обох моделей
    5. Створення та збереження візуалізації
    6. Збереження результатів у текстовий файл
    """

    # Створення директорії output, якщо вона не існує
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Створено директорію: {output_dir}")

    # Генерація імені файлу з поточною датою та часом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = os.path.join(output_dir, f"classification_comparison_{timestamp}.png")
    clusters_scatter_filename = os.path.join(output_dir, f"clusters_scatter_{timestamp}.png")
    clusters_simple_filename = os.path.join(output_dir, f"clusters_simple_{timestamp}.png")
    results_filename = os.path.join(output_dir, f"results_{timestamp}.txt")

    print("=" * 100)
    print("Порівняння алгоритмів класифікації цифр")
    print("=" * 100)

    # 1. Завантаження та підготовка даних
    print("\n[1/6] Завантаження датасету...")
    loader = DigitsDataLoader()
    X_train, X_test, y_train, y_test, X_scaled, y = loader.load_data()
    print(f"✓ Завантажено {len(X_scaled)} зразків")
    print(f"  - Тренувальна вибірка: {len(X_train)} зразків")
    print(f"  - Тестова вибірка: {len(X_test)} зразків")

    # 2. Логістична регресія (навчання з учителем)
    print("\n[2/6] Навчання моделі Logistic Regression...")
    log_reg = LogisticRegressionClassifier()
    log_reg.train(X_train, y_train)
    acc_lr = log_reg.accuracy(X_test, y_test)
    print(f"✓ Точність Logistic Regression: {acc_lr:.3f} ({acc_lr*100:.1f}%)")

    # 3. KMeans (навчання без учителя)
    print("\n[3/6] Навчання моделі K-Means...")
    kmeans = KMeansClusterer()
    kmeans.train(X_train)
    acc_km = kmeans.accuracy(X_train, y_train)
    print(f"✓ Приблизна точність K-Means: {acc_km:.3f} ({acc_km*100:.1f}%)")

    # Отримання міток кластерів для всіх даних
    kmeans_labels = kmeans.predict(X_scaled)

    # 4. Створення та збереження візуалізацій
    print("\n[4/6] Створення візуалізацій...")

    # 4.1 Комбінована візуалізація з межами
    DataVisualizer.plot_combined_boundaries(
        log_reg.model,
        kmeans.model,
        X_scaled,
        y,
        output_path=chart_filename
    )

    # 4.2 Детальна візуалізація кластерів
    DataVisualizer.plot_clusters_scatter(
        X_scaled,
        y,
        kmeans_labels,
        output_path=clusters_scatter_filename
    )

    # 4.3 Спрощена візуалізація кластерів (як на скріншоті)
    DataVisualizer.plot_simple_clusters(
        X_scaled,
        y,
        kmeans_labels,
        output_path=clusters_simple_filename
    )

    # 5. Збереження результатів у текстовий файл
    print("\n[5/6] Збереження результатів...")
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Результати порівняння алгоритмів класифікації цифр\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Дата та час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Інформація про датасет:\n")
        f.write(f"  - Загальна кількість зразків: {len(X_scaled)}\n")
        f.write(f"  - Тренувальна вибірка: {len(X_train)} зразків\n")
        f.write(f"  - Тестова вибірка: {len(X_test)} зразків\n")
        f.write(f"  - Кількість класів: 10 (цифри 0-9)\n")
        f.write(f"  - Кількість ознак після PCA: 2\n\n")

        f.write("Результати моделей:\n")
        f.write(f"  1. Logistic Regression (з учителем):\n")
        f.write(f"     Точність: {acc_lr:.4f} ({acc_lr*100:.2f}%)\n\n")
        f.write(f"  2. K-Means (без учителя):\n")
        f.write(f"     Точність: {acc_km:.4f} ({acc_km*100:.2f}%)\n\n")

        f.write("Висновки:\n")
        diff = acc_lr - acc_km
        f.write(f"  - Різниця в точності: {abs(diff):.4f} ({abs(diff)*100:.2f}%)\n")
        if diff > 0:
            f.write(f"  - Logistic Regression показала кращі результати на {diff*100:.2f}%\n")
        else:
            f.write(f"  - K-Means показав кращі результати на {abs(diff)*100:.2f}%\n")
        f.write("\nВізуалізація збережена у файл: " + chart_filename + "\n")

    print(f"✓ Результати збережено у: {results_filename}")

    print("\n[6/6] Генерація додаткових звітів...")
    print("✓ Всі візуалізації створено успішно!")

    print("\n" + "=" * 100)
    print("Виконання завершено успішно!")
    print("=" * 100)
    print(f"\nЗбережені файли:")
    print(f"  📊 Графік з межами: {chart_filename}")
    print(f"  📊 Детальні кластери: {clusters_scatter_filename}")
    print(f"  📊 Спрощені кластери: {clusters_simple_filename}")
    print(f"  📄 Результати: {results_filename}")


if __name__ == "__main__":
    main()
