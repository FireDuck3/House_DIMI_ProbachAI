# House Prices — навчена модель XGBoost

**Практична робота з машинного навчання:** прогнозування ціни будинку за 79 характеристиками. Датасет: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Завдання: **регресія, supervised learning**.

## Результати навчання

На реальному Kaggle `train.csv` (1460 рядків) перевірено моделі на відкладених 20% даних (292 рядки, `random_state=42`).

| Модель | Validation RMSE(log SalePrice) |
| --- | ---: |
| DummyRegressor (baseline) | 0.43325 |
| XGBoost (200 дерев) | 0.13196 |
| **XGBoost (600 дерев)** | **0.12657** |

Менша RMSE — краща. **Це валідація на навчальному датасеті, не Kaggle Leaderboard score.** Найкращий варіант повторно тренується на всіх 1460 рядках. Для 1459 тестових рядків формується `results/submission.csv`.

## Файли

- [train.py](train.py) — завантаження даних, підготовка ознак, навчання і порівняння двох XGBoost, перевірка точності, збереження моделі.
- [predict.py](predict.py) — прогнозування за допомогою збереженої моделі.
- [models/house_prices_xgboost.joblib](models/house_prices_xgboost.joblib) — натренована модель разом із preprocessing (створюється workflow або локальним навчанням).
- [results/metrics.json](results/metrics.json) — результати валідації.
- [results/submission.csv](results/submission.csv) — готові прогнози для Kaggle (створюється workflow).
- [requirements.txt](requirements.txt) — залежності.
- [.github/workflows/train-model.yml](.github/workflows/train-model.yml) — автоматичне навчання моделі у GitHub Actions.

## Як навчити модель самостійно

1. Завантажити `train.csv` і `test.csv` з Kaggle в папку `data/`.
2. У терміналі виконати:

```bash
python -m pip install -r requirements.txt
python train.py --data-dir data
```

Завантажити модель і зробити прогноз (повторне навчання не потрібне):

```bash
python predict.py --input data/test.csv --model models/house_prices_xgboost.joblib --output results/predictions.csv
```

## Автоматичне навчання

Workflow GitHub Actions отримує і **перевіряє по Git SHA-1** точні копії двох Kaggle CSV із [публічного репозиторію з даними](https://github.com/Rswcf/houseprices_kaggle) (вони побайтово збігаються з файлами, наданими для роботи). Далі запускає `train.py` і комітить модель та результати до цього репозиторію. Для повторного запуску: **Actions → Train House Prices XGBoost → Run workflow**.

**Використання joblib:** завантажуйте лише довірені моделі, оскільки відкриття .joblib може виконувати код. Для максимальної сумісності використовуйте версії бібліотек із `requirements.txt`.
