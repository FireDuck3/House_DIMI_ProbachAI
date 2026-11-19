"""Generate predictions using an already trained joblib pipeline."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, default=Path("data/test.csv"))
p.add_argument("--model", type=Path, default=Path("models/house_prices_xgboost.joblib"))
p.add_argument("--output", type=Path, default=Path("results/predictions.csv"))
args = p.parse_args()
model = joblib.load(args.model)  # Open only files from trusted sources.
test = pd.read_csv(args.input)
prices = np.exp(model.predict(test.drop(columns=["Id"])))
args.output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"Id": test["Id"], "SalePrice": prices}).to_csv(args.output, index=False)
print(f"Saved predictions to {args.output}")
