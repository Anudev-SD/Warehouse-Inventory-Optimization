# 📦 Inventory Management Optimization for Warehouse

A machine learning project that combines **deep learning** and **reinforcement learning** to improve warehouse inventory management:

- **LSTM** neural network to **forecast product demand** from historical sales
- **PPO reinforcement learning agent** to learn **stock placement** across warehouse shelves

> This project uses a **synthetic dataset** and a **simplified warehouse simulation**. It is a learning and portfolio project, not a production system.

## ✨ What this project does

1. **Data preparation**: aggregates sales by date and product, pivots into a time series, scales values to 0–1
2. **Demand forecasting**: builds sequences of 30 days and trains a 2-layer LSTM to predict the next day's demand for all products
3. **Stock placement**: creates a custom Gym warehouse environment and trains a PPO agent to assign items to shelves
4. **Visualization**: plots actual vs. forecasted demand and the final item-to-shelf placement

## 📊 Dataset

`synthetic_inventory_data.csv`: 18,250 rows × 10 columns (50 products over 365 days, starting 2023-01-01).

| Column | Description |
|---|---|
| `date` | Day of the record |
| `product_id` | Product identifier (1–50) |
| `product_category` | Category (e.g. Food, Clothing, Books, Furniture) |
| `sales` | Units sold that day (0–50) |
| `retrieval_frequency` | How often the item is picked (1–10) |
| `warehouse_location` | Aisle and shelf (e.g. `Aisle-8-Shelf-1`) |
| `stock_level` | Units in stock (20–200) |
| `restock_lead_time` | Days needed to restock (1–7) |
| `storage_constraints` | e.g. Fragile, Temperature-Sensitive |
| `price` | Item price |

## 🧠 Models

### 1. LSTM demand forecasting

```
Input (30 days × 50 products)
  → LSTM(50, return_sequences=True) → Dropout(0.2)
  → LSTM(50)                        → Dropout(0.2)
  → Dense(50)                       # next-day demand for each product
```

- Loss: Mean Squared Error, optimizer: Adam
- 20 epochs, batch size 32, 80/20 train/test split
- Result: training loss ≈ 0.096 and validation loss ≈ 0.089 (MSE on 0–1 scaled data)

### 2. PPO reinforcement learning for stock placement

A custom `WarehouseEnv` (OpenAI Gym) with:

- **State:** the shelf assigned to each of 10 items
- **Action:** choose one of 5 shelves
- **Reward:** penalty based on the distance between the item and the chosen shelf
- **Agent:** PPO (`stable-baselines3`, `MlpPolicy`), trained for 10,000 timesteps

Example learned placement (item → shelf): `[4 3 4 2 1 2 2 4 3 3]`

## 📁 Project structure

```
warehouse-inventory-optimization/
├── inventory_management_optimization_for_warehouse.ipynb
├── data/
│   └── synthetic_inventory_data.csv
├── requirements.txt
└── README.md
```

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Anudev-SD/warehouse-inventory-optimization.git
cd warehouse-inventory-optimization
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Point the notebook to the data

The notebook was written in Google Colab. Change the data path in the first cells from

```python
df = pd.read_csv('/content/drive/MyDrive/synthetic_inventory_data.csv')
```

to

```python
df = pd.read_csv('data/synthetic_inventory_data.csv')
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open the notebook and run all cells. You can also open it directly in Google Colab.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Gym · Stable-Baselines3 · matplotlib

## ⚠️ Limitations

- The data is synthetic, so the results don't reflect real warehouse behavior
- The RL environment is simplified: the reward is a toy distance penalty, and it doesn't yet use real constraints such as retrieval frequency, fragility, or temperature sensitivity
- The LSTM train/test split is random rather than chronological, and it is evaluated only with MSE on scaled data

## 🔮 Future improvements

- Use a chronological train/test split and add metrics like MAE and RMSE
- Compare the LSTM with baselines (moving average, ARIMA, Prophet)
- Build a more realistic RL reward using retrieval frequency, storage constraints, and travel distance
- Connect forecasts to reorder-point and safety-stock calculations
- Wrap the models in a small FastAPI or Streamlit app

## 📄 License

MIT. Feel free to use this project for learning.
