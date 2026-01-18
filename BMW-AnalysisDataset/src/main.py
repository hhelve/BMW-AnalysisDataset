import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import datetime
import numpy as np
from pathlib import Path


current_year = datetime.datetime.now().year

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "bmw.csv"

df = pd.read_csv(CSV_PATH)

y = np.log1p(df["price"])
X = df.drop("price", axis=1)

X["car_age"] = current_year - X["year"]
X.drop("year", axis=1, inplace=True)

X = pd.get_dummies(
    X,
    columns=["model", "transmission", "fuelType"],
)

train_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=4000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    objective="reg:squarederror",
    eval_metric="mae",
    random_state=42,
    min_child_weight=4,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
preds_log = model.predict(X_test)

preds = np.expm1(preds_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, preds)
print(mae)
modelos_validos = sorted(
    col.replace("model_", "") for col in train_columns if col.startswith("model_")
)

transmissoes_validas = sorted(
    col.replace("transmission_", "")
    for col in train_columns
    if col.startswith("transmission_")
)

combustiveis_validos = sorted(
    col.replace("fuelType_", "") for col in train_columns if col.startswith("fuelType_")
)

print("\nModelos disponíveis:")
for i, m in enumerate(modelos_validos):
    print(f"{i} - {m}")

print("\nTransmissões disponíveis:")
for i, t in enumerate(transmissoes_validas):
    print(f"{i} - {t}")

print("\nCombustíveis disponíveis:")
for i, f in enumerate(combustiveis_validos):
    print(f"{i} - {f}")

while True:
    try:
        year = int(input("\nAno de fabricação: "))
        mileage = int(input("Quilometragem: "))
        tax = int(input("Imposto: "))
        mpg = float(input("Consumo (mpg): "))

        idx_model = int(input("Escolha o índice do modelo: "))
        idx_trans = int(input("Escolha o índice da transmissão: "))
        idx_fuel = int(input("Escolha o índice do combustível: "))

        modelo = modelos_validos[idx_model]
        transmission = transmissoes_validas[idx_trans]
        fuel = combustiveis_validos[idx_fuel]

        break
    except (ValueError, IndexError):
        print("Entrada inválida. Tente novamente.")

new_car = pd.DataFrame(0, index=[0], columns=train_columns)

new_car["car_age"] = current_year - year
new_car["mileage"] = mileage
new_car["tax"] = tax
new_car["mpg"] = mpg

model_col = f"model_{modelo}"
trans_col = f"transmission_{transmission}"
fuel_col = f"fuelType_{fuel}"

new_car[model_col] = 1
new_car[trans_col] = 1
new_car[fuel_col] = 1

valor_log = model.predict(new_car)[0]
valor = np.expm1(valor_log)
print(f"\nO valor estimado do carro é {valor:,.2f} dolares.")
