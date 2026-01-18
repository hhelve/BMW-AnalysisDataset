## BMW Price Prediction with XGBoost

Projeto de **Machine Learning** para estimar o preço de carros BMW usando dados tabulares e **XGBoost**.

O modelo utiliza **transformação logarítmica no preço**, *one-hot encoding* para variáveis categóricas e engenharia de features com a idade do carro.

---

## Dataset
Arquivo `bmw.csv` contendo:
- model, year, price, mileage, tax, mpg, fuelType, transmission

---

## Modelo
- XGBoost Regressor
- Métrica: **MAE** (em valores reais)

---

## Execução

```bash
pip install -r requirements.txt
python main.py
