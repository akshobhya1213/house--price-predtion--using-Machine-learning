# 🏠 House Price Prediction (ML)

End-to-end machine-learning project: data loading → feature engineering → model comparison → hyper-parameter tuning → evaluation → saved model → CLI + REST API + web form.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train
python -m src.train --data my.csv
python -m src.predict --MedInc 6 --HouseAge 15 --Latitude 37.77 --Longitude -122.42
python app.py
pytest
```

## Pipeline
- Feature engineering: bedroom ratio, rooms/person, log population, distance to LA/SF
- Preprocessing: median imputation and standard scaling in a sklearn Pipeline
- Models: Linear, Ridge, Random Forest, Hist. Gradient Boosting
- Tuning: RandomizedSearchCV
- Outputs: models/model.joblib and reports/

Target `price` is in units of **$100,000**. Metrics in reports/ reflect the dataset noted in metrics.json.