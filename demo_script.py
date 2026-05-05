#   FUEL EFFICIENCY PREDICTOR — Auto MPG Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras

#  STEP 1: LOAD DATASET
def load_data():
    print("\n[1/5] Loading Auto MPG dataset...")
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    
    df = pd.read_csv(url, names=column_names, sep=r'\s+', na_values='?', comment='\t')
    
    print(f"    Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

#  STEP 2: CLEAN & PREPROCESS
def preprocess(df):
    print("\n[2/5] Cleaning and preprocessing data...")

    # Drop car name (not useful for prediction)
    df = df.drop(columns=['car_name'])

    # Drop rows with missing horsepower values
    missing = df['horsepower'].isna().sum()
    df = df.dropna()
    print(f"    Removed {missing} rows with missing horsepower values")

    # One-hot encode 'origin' (1=USA, 2=Europe, 3=Japan)[cite: 1]
    df['origin_USA']    = (df['origin'] == 1).astype(int)
    df['origin_Europe'] = (df['origin'] == 2).astype(int)
    df['origin_Japan']  = (df['origin'] == 3).astype(int)
    df = df.drop(columns=['origin'])
    print(f"    'Origin' column one-hot encoded into 3 columns")

    return df

#  STEP 3: VISUALIZE WITH SEABORN
def visualize(df):
    print("\n[3/5] Generating pairplot visualization...")
    selected = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
    sns.pairplot(selected, diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Auto MPG Dataset — Pairplot', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=100, bbox_inches='tight')
    print("    Pairplot saved as 'pairplot.png'")
    plt.close()

#  STEP 4: TRAIN/TEST SPLIT + SCALE
def split_and_scale(df):
    print("\n[4/5] Splitting and scaling data...")

    X = df.drop(columns=['mpg'])
    y = df['mpg']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Training samples : {len(X_train)}")
    print(f"    Testing  samples : {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


#  STEP 5: BUILD & TRAIN MODEL
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),  # Hidden layer 1
        keras.layers.Dense(64, activation='relu'),                               # Hidden layer 2
        keras.layers.Dense(1)                                                    # Output layer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(X_train, y_train):
    print("\n[5/5] Building and training the neural network...")
    model = build_model(X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        verbose=0
    )

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=100)
    print("    Training loss plot saved as 'training_loss.png'")
    plt.close()

    return model

#  EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    print("\n─── Model Evaluation ───────────────────────")
    predictions = model.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, predictions)
    mse  = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"    MAE  (Mean Absolute Error) : {mae:.2f} MPG")
    print(f"    MSE  (Mean Squared Error)  : {mse:.2f}")
    print(f"    RMSE (Root MSE)            : {rmse:.2f} MPG")
    print("────────────────────────────────────────────")

    # Actual vs Predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.title('Actual vs Predicted MPG')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=100)
    print("    Actual vs Predicted plot saved as 'actual_vs_predicted.png'")
    plt.close()

    return predictions


#  HARDCODED TEST CASES
def run_hardcoded_tests(model, scaler, feature_columns):
    print("\n─── Hardcoded Test Cases ────────────────────")

    test_vehicles = [
        {
            "name"        : "1973 Muscle Car (USA)",
            "cylinders"   : 8, "displacement": 350, "horsepower": 165,
            "weight"      : 4000, "acceleration": 12, "model_year": 73,
            "origin_USA"  : 1, "origin_Europe": 0, "origin_Japan": 0
        },
        {
            "name"        : "1978 Japanese Compact",
            "cylinders"   : 4, "displacement": 98, "horsepower": 75,
            "weight"      : 2100, "acceleration": 16, "model_year": 78,
            "origin_USA"  : 0, "origin_Europe": 0, "origin_Japan": 1
        },
        {
            "name"        : "1976 European Mid-size",
            "cylinders"   : 6, "displacement": 200, "horsepower": 110,
            "weight"      : 2800, "acceleration": 15, "model_year": 76,
            "origin_USA"  : 0, "origin_Europe": 1, "origin_Japan": 0
        },
        {
            "name"        : "1971 Heavy US Sedan",
            "cylinders"   : 8, "displacement": 400, "horsepower": 190,
            "weight"      : 4800, "acceleration": 10, "model_year": 71,
            "origin_USA"  : 1, "origin_Europe": 0, "origin_Japan": 0
        },
        {
            "name"        : "1980 Fuel-Efficient Hatchback",
            "cylinders"   : 4, "displacement": 85, "horsepower": 65,
            "weight"      : 1900, "acceleration": 18, "model_year": 80,
            "origin_USA"  : 0, "origin_Europe": 0, "origin_Japan": 1
        },
    ]

    for vehicle in test_vehicles:
        name = vehicle.pop("name")
        features = pd.DataFrame([vehicle])[feature_columns]
        features_scaled = scaler.transform(features)
        predicted_mpg = model.predict(features_scaled, verbose=0)[0][0]
        print(f"    {name:<35} → Predicted MPG: {predicted_mpg:.1f}")

    print("────────────────────────────────────────────")

#  USER INPUT PREDICTION
def get_user_input(model, scaler, feature_columns):
    print("\n─── Custom Vehicle Prediction ───────────────")
    print("    Enter your vehicle specs below:\n")

    try:
        cylinders    = int(input("    Cylinders (4 / 6 / 8)           : "))
        displacement = float(input("    Displacement (cu. in., e.g. 200) : "))
        horsepower   = float(input("    Horsepower (e.g. 130)            : "))
        weight       = float(input("    Weight (lbs, e.g. 3000)          : "))
        acceleration = float(input("    Acceleration (0-60 sec, e.g. 14) : "))
        model_year   = int(input("    Model year (e.g. 76 for 1976)    : "))
        origin_input = int(input("    Origin (1=USA, 2=Europe, 3=Japan): "))

        origin_USA    = 1 if origin_input == 1 else 0
        origin_Europe = 1 if origin_input == 2 else 0
        origin_Japan  = 1 if origin_input == 3 else 0

        features = pd.DataFrame([{
            'cylinders': cylinders, 'displacement': displacement,
            'horsepower': horsepower, 'weight': weight,
            'acceleration': acceleration, 'model_year': model_year,
            'origin_USA': origin_USA, 'origin_Europe': origin_Europe,
            'origin_Japan': origin_Japan
        }])[feature_columns]

        features_scaled = scaler.transform(features)
        predicted_mpg   = model.predict(features_scaled, verbose=0)[0][0]

        print(f"\n    Predicted Fuel Efficiency: {predicted_mpg:.1f} MPG")

        if predicted_mpg >= 30:
            print("    Rating: Efficient")
        elif predicted_mpg >= 20:
            print("    Rating: Average")
        else:
            print("    Rating: Low efficiency")

    except ValueError:
        print("    Invalid input. Please enter numbers only.")

    print("────────────────────────────────────────────")

#  MAIN
def main():
    print("   FUEL EFFICIENCY PREDICTOR — Auto MPG Neural Network")
    
    # Run all steps
    df                                              = load_data()
    df                                              = preprocess(df)
    visualize(df)
    X_train, X_test, y_train, y_test, scaler, cols = split_and_scale(df)
    model                                           = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    run_hardcoded_tests(model, scaler, cols)

    # Ask user if they want to enter their own vehicle
    print("\nWould you like to predict MPG for your own vehicle?")
    choice = input("Enter 'yes' to continue or any other key to exit: ").strip().lower()
    if choice == 'yes':
        get_user_input(model, scaler, cols)

    print("\nDone! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()