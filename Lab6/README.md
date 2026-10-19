# Лабораторна робота №6

## Тема

Нейронна регресія для табличних даних у TensorFlow.

## Мета

Підготувати реальні табличні дані без витоку інформації, порівняти нейромережу з класичними моделями, оцінити якість прогнозування MPG та зберегти весь процес прогнозування.

## Дані

Використано UCI Auto MPG.

- Ціль: `mpg` — милі на галон пального;
- більше значення `mpg` означає вищу паливну ефективність;
- джерело: https://archive.ics.uci.edu/dataset/9/auto+mpg;
- DOI: 10.24432/C5859H;
- ліцензія: CC BY 4.0;
- локальна копія: `auto-mpg.data`.

## Варіант 5

Порівняно базову нейромережу з двома шарами `Dense(32, relu)` та мережею з двома шарами `Dense(32, tanh)`. Інші налаштування не змінювалися.

## Виконано

- Дані поділено на train, validation і test у співвідношенні приблизно 70/15/15.
- Пропуски й категорію `origin` оброблено через `ColumnTransformer`, навчений лише на train.
- Побудовано DummyRegressor, LinearRegression і дві нейромережі.
- Нейромережі навчено з Normalization, EarlyStopping і ModelCheckpoint.
- Моделі обрано за validation RMSE, а test використано для фінальної оцінки.
- Побудовано графіки фактичних і прогнозованих значень, а також залишків.
- Збережено модель, препроцесор, metadata та таблиці результатів.

## Структура

- `Lab6_Vakhniuk.ipynb` — виконаний ноутбук;
- `auto-mpg.data` — локальна копія даних;
- `validation_results.csv` — результати на validation;
- `final_results.csv` — фінальні test-результати;
- `largest_errors.csv` — п’ять найбільших помилок;
- `requirements.txt` — версії бібліотек;
- `artifacts/lab6_model.keras` — краща нейромережа;
- `artifacts/preprocessor.joblib` — препроцесор;
- `artifacts/metadata.json` — відомості про експеримент.

## Відтворення

```bash
python -m pip install -r requirements.txt