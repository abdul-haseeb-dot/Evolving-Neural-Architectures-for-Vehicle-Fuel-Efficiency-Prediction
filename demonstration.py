"""
demonstration.py
================
Interactive CLI tool for predicting vehicle fuel efficiency (MPG).

HOW IT WORKS:
  - On first run: trains the Manual Neural Network [64, 32] using the same
    pipeline as model_training.py, then saves the model + scaler to disk.
  - On subsequent runs: loads the saved model + scaler instantly.
  - Prompts the user for raw vehicle specs (same scale as the original dataset).
  - Applies the identical preprocessing (feature engineering + scaling) and
    outputs the predicted MPG.
"""

import os
import sys
import pickle
import random

import numpy as np
import pandas as pd

# ── Suppress TensorFlow/Keras noise ──────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths for persisted artefacts ────────────────────────────────────────────
MODEL_PATH  = "demo_manual_nn.keras"
SCALER_PATH = "demo_scaler.pkl"
SEED        = 42

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COLOUR HELPERS  (pure ANSI – no external deps)
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def h(text, colour=CYAN):
    return f"{BOLD}{colour}{text}{RESET}"

def dim(text):
    return f"{DIM}{text}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN & SAVE  (runs once, on first launch)
# ─────────────────────────────────────────────────────────────────────────────

def build_manual_nn(input_dim):
    """Identical architecture to model_training.py: [64, 32]."""
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_save():
    """
    Fetches the Auto MPG data via data_prep.prepare_data(), trains the
    Manual Neural Network [64,32], and persists both the model and a fresh
    StandardScaler so single-sample inference works correctly.
    """
    print(f"\n{h('[ FIRST RUN ]', YELLOW)}  No saved model found.")
    print(f"{dim('Fetching data and training the Manual Neural Network [64, 32]...')}\n")

    # ── Import data_prep from the same directory ──────────────────────────
    try:
        from data_prep import prepare_data
    except ModuleNotFoundError:
        print(f"{RED}ERROR:{RESET}  data_prep.py not found. "
              "Place it in the same folder as demonstration.py and try again.")
        sys.exit(1)

    X_train_full, X_test, y_train_full, y_test, feature_names, _ = prepare_data()

    # Validation split — mirrors model_training.py exactly
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED
    )

    model = build_manual_nn(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
    )

    print("  Training ... ", end="", flush=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    epochs_run = len(history.history["loss"])
    best_loss  = min(history.history["val_loss"])
    print(f"done  ({epochs_run} epochs, best val_loss = {best_loss:.4f})")

    # ── Re-fit a StandardScaler on the raw training split ────────────────
    #    data_prep returns already-scaled numpy arrays, so we need to
    #    rebuild the scaler from scratch to be able to transform new inputs.
    url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/"
           "auto-mpg/auto-mpg.data")
    col_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
                 "Acceleration", "Model Year", "Origin"]
    df = pd.read_csv(url, names=col_names, na_values="?",
                     comment="\t", sep=" ", skipinitialspace=True)
    df["Horsepower"]   = df["Horsepower"].fillna(df["Horsepower"].median())
    df["Power_to_Weight"] = df["Horsepower"] / df["Weight"]
    df["Origin"]       = df["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    df = pd.get_dummies(df, columns=["Origin"], prefix="", prefix_sep="")

    X_full = df.drop("MPG", axis=1)
    y_full = df["MPG"]

    # Same 80/20 split and random_state as data_prep.py
    X_tr, _, _, _ = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_tr)

    # ── Persist artefacts ─────────────────────────────────────────────────
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump((scaler, list(X_full.columns)), f)

    print(f"  {GREEN}Model saved  →{RESET}  {MODEL_PATH}")
    print(f"  {GREEN}Scaler saved →{RESET}  {SCALER_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artefacts():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 4. INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_float(label, lo, hi, unit=""):
    """Ask for a numeric value, validate range, loop until valid."""
    unit_str = f" {unit}" if unit else ""
    while True:
        raw = input(
            f"  {BOLD}{label}{RESET}{unit_str}  "
            f"{dim(f'[{lo} – {hi}]')}: "
        ).strip()
        if raw.lower() in ("q", "quit"):
            raise SystemExit
        try:
            val = float(raw)
        except ValueError:
            print(f"    {RED}Not a valid number. Try again.{RESET}")
            continue
        if not (lo <= val <= hi):
            print(f"    {RED}Out of range ({lo}–{hi}). Try again.{RESET}")
            continue
        return val


def prompt_origin():
    """Display numbered origin choices and return the selected string."""
    choices = [
        ("USA",    "USA     (North America)"),
        ("Europe", "Europe  (Germany, France, Italy …)"),
        ("Japan",  "Japan   (Toyota, Honda, Mazda …)"),
    ]
    print(f"\n  {BOLD}Region of Origin{RESET}")
    for idx, (_, desc) in enumerate(choices, 1):
        print(f"    {dim(str(idx))}  {desc}")
    while True:
        raw = input(f"  Enter choice {dim('[1–3]')}: ").strip()
        if raw.lower() in ("q", "quit"):
            raise SystemExit
        try:
            n = int(raw)
            if 1 <= n <= 3:
                return choices[n - 1][0]
        except ValueError:
            pass
        print(f"    {RED}Enter 1, 2, or 3.{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPROCESS A SINGLE USER INPUT
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_input(cylinders, displacement, horsepower, weight,
                     acceleration, model_year, origin, scaler, feature_names):
    """
    Mirrors data_prep.py exactly:
      1. Derive Power-to-Weight ratio
      2. One-hot encode Origin  (Europe / Japan / USA columns)
      3. Reorder columns to match training feature order
      4. Scale with the saved StandardScaler
    """
    row = {
        "Cylinders":        cylinders,
        "Displacement":     displacement,
        "Horsepower":       horsepower,
        "Weight":           weight,
        "Acceleration":     acceleration,
        "Model Year":       model_year,
        "Power_to_Weight":  horsepower / weight,
        "Europe":           1 if origin == "Europe" else 0,
        "Japan":            1 if origin == "Japan"  else 0,
        "USA":              1 if origin == "USA"    else 0,
    }
    df_row = pd.DataFrame([row])[feature_names]   # enforce exact column order
    return scaler.transform(df_row)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RESULT CARD
# ─────────────────────────────────────────────────────────────────────────────

def print_result(pred_mpg):
    bar_len  = 38
    bar_fill = int(min(pred_mpg / 50.0, 1.0) * bar_len)   # 0–50 mpg mapped to bar
    bar      = "█" * bar_fill + "░" * (bar_len - bar_fill)

    if pred_mpg >= 35:
        label = f"{GREEN}Excellent fuel economy 🌿{RESET}"
    elif pred_mpg >= 25:
        label = f"{CYAN}Good fuel economy{RESET}"
    elif pred_mpg >= 18:
        label = f"{YELLOW}Average fuel economy{RESET}"
    else:
        label = f"{RED}Below-average fuel economy{RESET}"

    print(f"""
{BOLD}{GREEN}╔════════════════════════════════════════╗
║       PREDICTED FUEL EFFICIENCY        ║
╠════════════════════════════════════════╣
║                                        ║
║          {pred_mpg:>6.2f}  MPG                   ║
║                                        ║
║ {bar} ║
║ {dim('Low ◄──────────────────────────► High')}  ║
╚════════════════════════════════════════╝{RESET}

  {label}
""")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SINGLE PREDICTION ROUND
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    # (name,          low,   high,  unit)
    ("Cylinders",     3,     8,     ""),
    ("Displacement",  68,    455,   "cu. in."),
    ("Horsepower",    46,    230,   "hp"),
    ("Weight",        1613,  5140,  "lbs"),
    ("Acceleration",  8.0,   25.0,  "sec  (0 → 60 mph)"),
    ("Model Year",    70,    82,    "(e.g. 76 = 1976)"),
]


def predict_once(model, scaler, feature_names):
    print(f"\n{h('─── Car Specifications ──────────────────────────────────────', DIM)}\n")
    values = {}
    for name, lo, hi, unit in FIELDS:
        values[name] = prompt_float(name, lo, hi, unit)

    origin = prompt_origin()

    X = preprocess_input(
        cylinders    = values["Cylinders"],
        displacement = values["Displacement"],
        horsepower   = values["Horsepower"],
        weight       = values["Weight"],
        acceleration = values["Acceleration"],
        model_year   = values["Model Year"],
        origin       = origin,
        scaler       = scaler,
        feature_names= feature_names,
    )

    pred_mpg = float(model.predict(X, verbose=0)[0][0])
    print_result(pred_mpg)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

BANNER = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗
║   Vehicle Fuel Efficiency Predictor  —  Manual Neural Net [64,32]║
╚══════════════════════════════════════════════════════════════════╝{RESET}

  Enter your car's specs on the {BOLD}original raw-data scale{RESET}.
  The model will predict fuel efficiency in {BOLD}Miles Per Gallon (MPG){RESET}.

  Type  {BOLD}q{RESET}  at any prompt to quit.
"""


def main():
    # ── Train once if no saved model ──────────────────────────────────────
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        train_and_save()
    else:
        print(f"\n{GREEN}✔{RESET}  Loaded saved model: {dim(MODEL_PATH)}")

    model, scaler, feature_names = load_artefacts()
    print(BANNER)

    # ── Prediction loop ───────────────────────────────────────────────────
    while True:
        predict_once(model, scaler, feature_names)

        again = input(
            f"  {BOLD}Predict another car?{RESET}  {dim('[y / n]')}: "
        ).strip().lower()
        if again not in ("y", "yes"):
            break

    print(f"\n{dim('Goodbye! Thanks for using the MPG Predictor.')}\n")


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        print(f"\n{dim('Exiting. Goodbye!')}\n")