# Chicago Housing Price Prediction 🏡📈

This project builds a machine learning pipeline using **R** and **XGBoost** to predict housing sale prices in Chicago based on property characteristics and geographic data. The model performs feature engineering, data transformation, and hyperparameter-tuned training using cross-validation. Final predictions are generated for a future dataset and exported for downstream analysis.

---

## 🧠 Project Goals

- Clean and preprocess property data
- Train a predictive model to estimate log-transformed sale prices
- Use XGBoost with cross-validation to optimize performance
- Generate predictions for unseen properties

---

## 🔧 Tools & Libraries

- **R** (Data processing & modeling)
- `readr`, `dplyr`, `xgboost`
- Cross-validation and RMSE evaluation
- Feature importance visualization

---

## 📁 File Structure
Chicago-Housing-Predict/ ├── Chicago-Housing-Predict.R 
> Note: `data/` folder and `.csv` files are excluded for privacy reasons.

---

## 🔍 Key Features & Steps

### 🔹 Data Preparation
- Drop columns with too many missing values
- Omit incomplete rows
- Log-transform skewed features (e.g., `sale_price`, `char_hd_sf`)
- Normalize numeric columns
- Encode categorical columns as integers
- Create new feature: `room_bed_ratio`

### 🔹 Modeling
- Use **XGBoost** (`reg:squarederror`)
- 5-fold cross-validation
- Optimal number of rounds via early stopping
- Evaluate model performance using RMSE and normalized RMSE

### 🔹 Prediction Pipeline
- Clean new dataset with the same transformations
- Predict prices using the trained model
- Reverse log transformation and export final results

---

## 📊 Example Output

- RMSE on test set: ~0.19  
- Normalized RMSE: ~14.7%  
- Output CSV: `final_predictions.csv` (not included)

---

## 📈 Feature Importance

The model includes automatic plotting of feature importance using `xgb.plot.importance()` to explain the top drivers of sale price.

---

## ⚠️ Notes

- **Input data** (`historic_property_data.csv`, `predict_property_data.csv`) is not shared due to privacy and size.
- Paths are currently hardcoded; adjust if you want to run in your environment.

---

## 👤 Author

**Yun-Shan Chung**  
[GitHub](https://github.com/a0828451) 

> Feel free to fork this repo, or reach out if you're working with housing data too!

