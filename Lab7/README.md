# Лабораторна робота №7

Нейронна класифікація зображень у TensorFlow.

Студент: Михайло Вахнюк.
Варіант: 5.

## Дані

Fashion-MNIST: зображення одягу 28×28 пікселів у відтінках сірого,
10 класів.

Джерело: https://github.com/zalandoresearch/fashion-mnist
Ліцензія: MIT.

Дані отримуються автоматично через:
keras.datasets.fashion_mnist.load_data()

Повна копія датасету до репозиторію не додається.

Поділ: 48 000 train, 12 000 validation, 10 000 test.
SEED = 42. Для train/validation використано стратифікацію.

## Моделі

- DummyClassifier(strategy="most_frequent").
- Базова Dense-мережа: Adam, learning_rate=0.001.
- Варіант 5: та сама мережа, learning_rate=0.0003.

Архітектура: Rescaling → Flatten → Dense(128, ReLU)
→ Dense(10, softmax).

Масштабування пікселів із 0–255 до 0–1 включене в модель.
Мітки — цілі індекси 0–9, функція втрат —
SparseCategoricalCrossentropy(from_logits=False).

Спільні налаштування: batch_size=128, максимум 30 епох,
EarlyStopping(patience=4), checkpoint за val_loss.

## Вибір моделі

Критерій: найменша validation cross-entropy.
За рівності обирається базова мережа.
Обрана модель: Варіант 5 (lr=0.0003).

Test використано для фінального оцінювання після вибору моделі.

## Файли

- Lab7_Vakhniuk.ipynb — виконаний ноутбук.
- requirements.txt — версії бібліотек.
- model_comparison.csv — порівняння моделей.
- classification_report.csv — метрики за класами.
- confident_errors.csv — найвпевненіші помилки, якщо вони є.
- class_examples.png — приклади класів.
- training_curves.png — графіки навчання.
- confusion_matrix.png — матриця помилок.
- confident_errors.png — приклади помилок, якщо вони є.
- class_scores.png — оцінки класів для окремих прикладів.
- artifacts/base_best.keras — найкраща базова мережа.
- artifacts/variant5_best.keras — найкраща варіантна мережа.
- artifacts/lab7_model.keras — обрана нейромережа.
- artifacts/metadata.json — порядок класів і параметри експерименту.
- artifacts/reload_probe.npz — три validation-приклади
  та прогнози для перевірки відновлення.

## Запуск

Встановити залежності:
python -m pip install -r requirements.txt

Відкрити ноутбук, вибрати відповідне Python-середовище
та виконувати комірки послідовно.
Робоча папка ядра має бути Lab7.

Для першого завантаження Fashion-MNIST потрібен інтернет.
Комірка перевірки відновлення може виконуватися окремо
після перезапуску ядра за наявності збережених артефактів.
