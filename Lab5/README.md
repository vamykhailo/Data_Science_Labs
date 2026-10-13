# Лабораторна робота №5

## Тема

Вступ до TensorFlow і побудова простої нейронної мережі для регресії.

## Мета

Створити тензори TensorFlow, побудувати та навчити нейронну мережу для регресії, оцінити її на незалежній вибірці й зберегти кращу модель у форматі `.keras`.

## Виконано

- Перевірено TensorFlow і виконано операції з тензорами.
- Згенеровано нелінійні дані та поділено їх на train, validation і test у співвідношенні 70/15/15.
- Побудовано baseline-модель із двома шарами `Dense(16, relu)`.
- Виконано варіант 5: створено модель із `tanh` у прихованих шарах.
- Порівняно моделі за мінімальним `val_loss`, test MSE і test MAE.
- Збережено кращу модель у `lab5_model.keras` та перевірено її повторне завантаження.

## Файли

- `Lab5_Vakhniuk.ipynb` — виконаний Jupyter Notebook;
- `lab5_model.keras` — збережена краща модель.

## Джерела

- https://www.tensorflow.org/guide/tensor
- https://www.tensorflow.org/tutorials/keras/regression
- https://www.tensorflow.org/guide/keras/training_with_built_in_methods

## Репозиторій

https://github.com/vamyvkhailo/Data_Science_Labs