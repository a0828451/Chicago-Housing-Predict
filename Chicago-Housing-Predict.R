# Load required libraries
library(readr)
library(dplyr)
library(xgboost)

# Step 1: Read the training data
data <- read_csv("C:/Users/a0828/OneDrive/桌面/bigData_final/data/historic_property_data.csv")

# Step 2: Clean the training data
# Remove columns with too many NAs
na_count <- colSums(is.na(data))
data_clean <- data[, na_count <= 200]
data_clean <- na.omit(data_clean)

# Drop irrelevant columns
data_clean <- data_clean %>% 
  select(-c(geo_property_zip))

# Log-transform skewed continuous features
data_clean <- data_clean %>%
  mutate(
    sale_price = log(sale_price + 1),
    char_hd_sf = log(char_hd_sf + 1),
    char_bldg_sf = log(char_bldg_sf + 1),
    geo_tract_pop = log(geo_tract_pop + 1)
  )

# Scale numeric features
data_clean <- data_clean %>%
  mutate(across(c(
    econ_tax_rate, char_age, geo_white_perc, geo_black_perc, geo_asian_perc
  ), scale))

# Add interaction feature
data_clean <- data_clean %>%
  mutate(room_bed_ratio = char_rooms / (char_beds + 1))

# Convert categorical variables to integer indices
data_clean <- data_clean %>%
  mutate(across(c(
    char_air, char_attic_type, char_cnst_qlty, geo_property_city, 
    geo_school_elem_district, geo_school_hs_district
  ), as.factor)) %>%
  mutate(across(where(is.factor), as.integer))

# Ensure all remaining columns are numeric
data_clean <- data_clean %>%
  mutate(across(everything(), as.numeric))

# Step 3: Split the dataset
set.seed(123)
train_indices <- sample(1:nrow(data_clean), 0.8 * nrow(data_clean))
train_data <- data_clean[train_indices, ]
test_data <- data_clean[-train_indices, ]

# Prepare data for XGBoost
train_X <- as.matrix(train_data[, colnames(train_data) != "sale_price"])
train_y <- train_data$sale_price

test_X <- as.matrix(test_data[, colnames(test_data) != "sale_price"])
test_y <- test_data$sale_price

dtrain <- xgb.DMatrix(data = train_X, label = train_y)

# Step 4: Perform Cross-Validation
set.seed(123)
cv_results <- xgb.cv(
  data = dtrain,
  nrounds = 200,
  max_depth = 5,
  eta = 0.1,
  objective = "reg:squarederror",
  nfold = 5,
  verbose = TRUE,
  metrics = "rmse",
  early_stopping_rounds = 10
)

# Extract the best number of rounds
best_nrounds <- cv_results$best_iteration
print(paste("Best number of rounds from CV:", best_nrounds))

# Step 5: Train the Final XGBoost Model
xgb_model <- xgboost(
  data = dtrain,
  max_depth = 5, eta = 0.1, nrounds = best_nrounds, 
  objective = "reg:squarederror", verbose = FALSE
)

# Step 6: Predict on the Test Set
dtest <- xgb.DMatrix(data = test_X)
test_predictions <- predict(xgb_model, newdata = dtest)

# Evaluate Model Performance
rmse_value <- sqrt(mean((test_y - test_predictions)^2))
print(paste("RMSE (Test Set):", rmse_value))

# Calculate Normalized RMSE
mean_sale_price <- mean(test_y)
normalized_rmse <- (rmse_value / mean_sale_price) * 100
print(paste("Normalized RMSE:", round(normalized_rmse, 2), "%"))

# Step 7: Read the new data for prediction
future_data <- read_csv("C:/Users/a0828/OneDrive/桌面/bigData_final/data/predict_property_data.csv")

# Step 8: Clean the new data (similar to training data cleaning)
na_count_future <- colSums(is.na(future_data))
future_data_clean <- future_data[, na_count_future <= 200]
future_data_clean <- na.omit(future_data_clean)

# Drop irrelevant columns
future_data_clean <- future_data_clean %>%
  select(-c(geo_property_zip))

# Apply the same transformations
future_data_clean <- future_data_clean %>%
  mutate(
    char_hd_sf = log(char_hd_sf + 1),
    char_bldg_sf = log(char_bldg_sf + 1),
    geo_tract_pop = log(geo_tract_pop + 1),
    room_bed_ratio = char_rooms / (char_beds + 1)
  ) %>%
  mutate(across(c(
    char_air, char_attic_type, char_cnst_qlty, geo_property_city, 
    geo_school_elem_district, geo_school_hs_district
  ), as.factor)) %>%
  mutate(across(where(is.factor), as.integer)) %>%
  mutate(across(everything(), as.numeric))

# Extract `pid` column and remove it from predictors
pids <- future_data_clean$pid
future_X <- as.matrix(future_data_clean[, colnames(future_data_clean) != "pid"])

# Step 9: Predict Sale Prices for New Data
dfuture <- xgb.DMatrix(data = future_X)
future_predictions <- predict(xgb_model, newdata = dfuture)

# Step 10: Save Predictions
final_predictions <- data.frame(
  pid = pids,
  predicted_sale_price = exp(future_predictions)- 1  # Exponentiate to reverse log-transform
)

write.csv(final_predictions, "C:/Users/a0828/OneDrive/桌面/bigData_final/data/final_predictions.csv", row.names = FALSE)

print("Predictions saved to 'final_predictions.csv'")
file.exists("C:/Users/a0828/OneDrive/桌面/bigData_final/data/final_predictions.csv")

mean(data$sale_price)  # Historical average
mean(exp(future_predictions))  # Predicted average

importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)