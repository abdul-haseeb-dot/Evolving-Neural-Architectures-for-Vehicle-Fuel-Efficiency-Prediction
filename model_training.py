import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import prepare_data


# =========================================================
# 1. SETUP
# =========================================================

os.makedirs("results", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 2. METRIC FUNCTION
# =========================================================

def evaluate_model(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }


# =========================================================
# 3. BUILD NEURAL NETWORK
# =========================================================

def build_nn(input_dim, architecture):
    """
    architecture example: [64, 32]
    Means:
    Dense(64) -> Dense(32) -> Dense(1)
    """
    model = Sequential()

    model.add(Dense(architecture[0], activation="relu", input_shape=(input_dim,)))

    for neurons in architecture[1:]:
        model.add(Dense(neurons, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# =========================================================
# 4. TRAIN ONE ARCHITECTURE
# =========================================================

def train_architecture(X_train, y_train, X_val, y_val, architecture, epochs=150, verbose=0):
    model = build_nn(X_train.shape[1], architecture)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=verbose
    )

    val_loss = min(history.history["val_loss"])

    return model, history, val_loss


# =========================================================
# 5. GENETIC ALGORITHM HELPERS
# =========================================================

def random_architecture():
    """
    Generates a 2-hidden-layer neural network architecture.
    Example: [64, 32]
    """
    possible_neurons = [8, 16, 32, 64, 128]
    return [
        random.choice(possible_neurons),
        random.choice(possible_neurons)
    ]


def crossover(parent1, parent2):
    """
    Combines two parent architectures.
    """
    child = [
        random.choice([parent1[0], parent2[0]]),
        random.choice([parent1[1], parent2[1]])
    ]
    return child


def mutate(architecture, mutation_rate=0.3):
    """
    Randomly changes neuron count in one layer.
    """
    possible_neurons = [8, 16, 32, 64, 128]
    new_arch = architecture.copy()

    for i in range(len(new_arch)):
        if random.random() < mutation_rate:
            new_arch[i] = random.choice(possible_neurons)

    return new_arch


def run_genetic_algorithm(X_train, y_train, X_val, y_val, population_size=10, generations=5):
    population = [random_architecture() for _ in range(population_size)]

    ga_history = []
    best_model = None
    best_architecture = None
    best_val_loss = float("inf")
    best_training_history = None

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")

        scored_population = []

        for arch in population:
            print(f"Training architecture: {arch}")

            model, history, val_loss = train_architecture(
                X_train,
                y_train,
                X_val,
                y_val,
                arch,
                epochs=120,
                verbose=0
            )

            scored_population.append((arch, val_loss, model, history))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_architecture = arch
                best_model = model
                best_training_history = history

        scored_population.sort(key=lambda x: x[1])

        generation_best_arch = scored_population[0][0]
        generation_best_loss = scored_population[0][1]

        ga_history.append({
            "Generation": gen + 1,
            "Best Architecture": str(generation_best_arch),
            "Best Validation MSE": generation_best_loss
        })

        print(f"Best architecture this generation: {generation_best_arch}")
        print(f"Best validation MSE this generation: {generation_best_loss:.4f}")

        # Select top parents
        parents = scored_population[:4]

        # Elitism: keep best 2
        new_population = [parents[0][0], parents[1][0]]

        # Generate children
        while len(new_population) < population_size:
            p1 = random.choice(parents)[0]
            p2 = random.choice(parents)[0]

            child = crossover(p1, p2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    return best_model, best_architecture, best_val_loss, best_training_history, pd.DataFrame(ga_history)


# =========================================================
# 6. PLOTS
# =========================================================

def plot_loss_curve(history, file_name, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, file_name, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_ga_history(ga_df):
    plt.figure(figsize=(8, 5))
    plt.plot(ga_df["Generation"], ga_df["Best Validation MSE"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Validation MSE")
    plt.title("Genetic Algorithm Progress")
    plt.tight_layout()
    plt.savefig("results/ga_progress.png")
    plt.close()


# =========================================================
# 7. MAIN PIPELINE
# =========================================================

def main():
    print("Loading prepared data...")

    X_train_full, X_test, y_train_full, y_test, feature_names, full_df = prepare_data()

    # Important: validation split is created only from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=SEED
    )

    results = []

    # =====================================================
    # Linear Regression Baseline
    # =====================================================

    print("\nTraining Linear Regression baseline...")

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    linear_preds = linear_model.predict(X_test)

    results.append(
        evaluate_model("Linear Regression", y_test, linear_preds)
    )

    plot_actual_vs_predicted(
        y_test.values,
        linear_preds,
        "results/linear_actual_vs_predicted.png",
        "Linear Regression: Actual vs Predicted MPG"
    )

    # =====================================================
    # Manual Neural Network
    # =====================================================

    print("\nTraining Manual Neural Network...")

    manual_architecture = [64, 32]

    manual_model, manual_history, manual_val_loss = train_architecture(
        X_train,
        y_train,
        X_val,
        y_val,
        manual_architecture,
        epochs=150,
        verbose=0
    )

    manual_preds = manual_model.predict(X_test).flatten()

    results.append(
        evaluate_model("Manual Neural Network [64, 32]", y_test, manual_preds)
    )

    plot_loss_curve(
        manual_history,
        "results/manual_nn_loss_curve.png",
        "Manual Neural Network Loss Curve"
    )

    plot_actual_vs_predicted(
        y_test.values,
        manual_preds,
        "results/manual_nn_actual_vs_predicted.png",
        "Manual Neural Network: Actual vs Predicted MPG"
    )

    # =====================================================
    # Genetic Algorithm Neural Network
    # =====================================================

    print("\nRunning Genetic Algorithm for Neural Architecture Search...")

    best_ga_model, best_architecture, best_val_loss, best_ga_history, ga_df = run_genetic_algorithm(
        X_train,
        y_train,
        X_val,
        y_val,
        population_size=10,
        generations=5
    )

    ga_preds = best_ga_model.predict(X_test).flatten()

    results.append(
        evaluate_model(f"GA-Evolved Neural Network {best_architecture}", y_test, ga_preds)
    )

    plot_loss_curve(
        best_ga_history,
        "results/ga_best_nn_loss_curve.png",
        "GA-Evolved Neural Network Loss Curve"
    )

    plot_actual_vs_predicted(
        y_test.values,
        ga_preds,
        "results/ga_nn_actual_vs_predicted.png",
        "GA-Evolved Neural Network: Actual vs Predicted MPG"
    )

    plot_ga_history(ga_df)

    # Save GA history
    ga_df.to_csv("results/ga_history.csv", index=False)

    # Save final metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison_metrics.csv", index=False)

    # Save best model
    best_ga_model.save("results/best_ga_model.keras")

    print("\nFinal Model Comparison:")
    print(results_df)

    print("\nBest GA Architecture:", best_architecture)
    print("Best GA Validation MSE:", best_val_loss)

    print("\nSaved outputs inside results/ folder.")


if __name__ == "__main__":
    main()