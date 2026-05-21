# Evolving Neural Architectures for Vehicle Fuel Efficiency Prediction

A complete machine learning pipeline for predicting vehicle fuel efficiency (MPG) using the UCI Auto MPG Dataset. The project benchmarks three models: Linear Regression, a manually designed Neural Network, and a GA-evolved Neural Network, and includes an Evolutionary Neural Architecture Search (ENAS) framework built on a custom Genetic Algorithm.

---

## Overview

Fuel efficiency is one of the most practically important metrics in the automotive world. It affects consumer costs, regulatory compliance, and environmental impact all at once. Yet predicting it accurately is non-trivial because it emerges from complex, non-linear interactions between a vehicle's physical specs: engine displacement, weight, horsepower, model year, and region of manufacture. Traditional physics-based approaches require detailed thermodynamic parameters that are rarely public, so a data-driven approach is both more accessible and more scalable.

What makes this project go beyond a standard regression task is the use of evolutionary computation to automate architecture selection. Rather than manually tuning the neural network topology, we implemented a Genetic Algorithm that searches the space of possible architectures the same way natural selection searches the space of organisms: through variation, selection, and inheritance. This places the project at the intersection of predictive modelling and meta-learning, making it a simplified but fully functional instance of Neural Architecture Search (NAS).

The project covers:

1. Data preprocessing and feature engineering pipeline
2. Exploratory Data Analysis (EDA) to justify modelling decisions
3. Linear Regression baseline
4. Manually designed Sequential Neural Network
5. Genetic Algorithm for automated Neural Architecture Search
6. CLI interface for real-time MPG prediction

---

## Dataset

Source: [UCI Auto MPG Dataset](https://doi.org/10.24432/C5859H)

- 398 records of automobile models from 1970 to 1982
- 3 manufacturing regions: USA, Japan, Europe
- Target variable: MPG (Miles Per Gallon)

| Feature | Type | Role |
|---|---|---|
| MPG | Continuous | Target |
| Cylinders | Discrete | Predictor |
| Displacement | Continuous | Predictor |
| Horsepower | Continuous | Predictor |
| Weight | Continuous | Predictor |
| Acceleration | Continuous | Predictor |
| Model Year | Discrete | Predictor |
| Origin | Categorical | Predictor |

> The dataset has 6 missing values in the `Horsepower` column (encoded as `?` in the raw file) which are handled during preprocessing.

---

## Project Structure

```
.
├── data_prep.py        # Data preprocessing and feature engineering pipeline
├── eda_analysis.py     # Exploratory Data Analysis and visualisations
├── model_training.py   # Model training and Genetic Algorithm for NAS
└── demonstration.py    # CLI interface for real-time MPG prediction
```

`data_prep.py` handles everything from raw data ingestion to a clean, scaled, split-ready dataset. `eda_analysis.py` runs the visual analysis and generates all EDA plots. `model_training.py` trains all three models and runs the full GA search. `demonstration.py` is the end-user facing script that accepts vehicle specs and returns a live MPG prediction.

---

## Installation

Make sure you have Python 3.8+ installed.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

Dependencies:

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
openpyxl
```

---

## How to Run

Run each script in order:

### 1. Data Preprocessing
```bash
python data_prep.py
```
Fetches the dataset, handles missing values, engineers the Power-to-Weight feature, one-hot encodes Origin, scales features, and exports the cleaned data.

### 2. Exploratory Data Analysis
```bash
python eda_analysis.py
```
Generates and saves three visualisations: pairplot, correlation heatmap, and origin boxplot.

### 3. Model Training and Genetic Algorithm
```bash
python model_training.py
```
Trains all three models (Linear Regression, Manual NN, GA-evolved NN), runs the Genetic Algorithm for 5 generations, and saves evaluation plots and results.

### 4. Real-Time CLI Prediction
```bash
python demonstration.py
```
Interactive prompt where you enter vehicle specs and get an instant MPG prediction from the best-performing model.

---

## Models

### Linear Regression (Baseline)
- `scikit-learn` `LinearRegression`
- Fits a hyperplane via least squares
- Serves as the performance floor

### Manual Neural Network
- Architecture: [64, 32] (two hidden layers)
- Activation: ReLU on hidden layers, linear on output
- Optimiser: Adam | Loss: MSE
- Trained for up to 150 epochs, batch size 16
- EarlyStopping with patience=15, restoring best weights

### GA-Evolved Neural Network
- Architecture discovered by the Genetic Algorithm: [8, 128]
- Same training procedure as the manual NN (up to 120 epochs)

---

## Genetic Algorithm

The GA treats each neural network architecture as an individual and evolves them across generations.

| Parameter | Value |
|---|---|
| Search space per layer | {8, 16, 32, 64, 128} |
| Total possible architectures | 25 |
| Population size | 10 |
| Generations | 5 |
| Parents selected | Top 4 by val MSE |
| Elitism | Best 2 preserved per generation |
| Mutation probability | 0.3 per layer |
| Fitness function | Validation MSE (lower = better) |

Crossover: each layer independently inherits from either parent with 50% probability.  
Mutation: each layer independently replaced with a random value from the search set with 30% probability.

The best architecture [8, 128] was found in Generation 1, a narrow bottleneck layer followed by a wide expansion layer, which interestingly mirrors patterns seen in autoencoder design.

---

## Results

### Test Set Performance

| Model | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| Linear Regression | 8.03 | 2.83 | 2.28 | 0.851 |
| Manual NN [64, 32] | 4.77 | 2.19 | 1.76 | 0.911 |
| GA-Evolved [8, 128] | 5.21 | 2.28 | 1.72 | 0.903 |

Neural networks outperform linear regression by a large margin with a 40.6% reduction in MSE. The manual NN edges out the GA-evolved model on R2 and RMSE, while the GA-evolved model achieves a slightly lower MAE, meaning it makes fewer medium-sized errors at the cost of occasional larger deviations.

### GA Progress by Generation

| Generation | Best Architecture | Best Val. MSE |
|---|---|---|
| 1 | [8, 128] | 8.917 |
| 2 | [128, 128] | 8.983 |
| 3 | [128, 128] | 9.425 |
| 4 | [128, 128] | 9.351 |
| 5 | [128, 128] | 9.328 |

The non-monotonic curve is expected behaviour for small-population GAs. The best solution was found early and preserved via elitism throughout.

---

## Visualisations

All plots are saved automatically when running `eda_analysis.py` and `model_training.py`.

### Feature Correlation Matrix

<img width="500" height="400" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/e22778c0-0ca4-485e-98c7-e5c90ae52f7d" />

Weight (r = -0.83) and Horsepower (r = -0.77) are the strongest predictors of MPG. The high Horsepower-Weight correlation (r = 0.86) motivated the Power-to-Weight ratio as a composite feature.

### Fuel Efficiency by Region

<img width="400" height="300" alt="origin_boxplot" src="https://github.com/user-attachments/assets/1f35b854-bd46-4f85-aace-68ac53126508" />

Japanese vehicles average around 31 MPG, European around 27 MPG, and American around 18-19 MPG. A 12 MPG median gap between USA and Japan justifies including Origin as a one-hot encoded feature.

### Pairplot

<img width="614" height="614" alt="pairplot_audit" src="https://github.com/user-attachments/assets/f677322c-4c76-4bc7-a1ab-038b06f8d04c" />

The curved (hyperbolic) scatter between MPG and both Horsepower and Weight is the primary visual justification for using neural networks over linear models.

### Actual vs. Predicted

| Linear Regression | Manual NN | GA-Evolved NN |
|---|---|---|
| <img width="300" height="300" alt="linear_actual_vs_predicted" src="https://github.com/user-attachments/assets/351c376e-46ba-4c7c-9aca-15c7eba9dec1" /> | <img width="300" height="300" alt="manual_nn_actual_vs_predicted" src="https://github.com/user-attachments/assets/c16919b3-8b76-4580-9e9c-811d7a5944f2" /> | <img width="300" height="300" alt="ga_nn_actual_vs_predicted" src="https://github.com/user-attachments/assets/ea4f8076-eafb-4945-ae0e-2e57b28ad28d" /> |

### Loss Curves

| Manual NN | GA-Evolved NN |
|---|---|
| <img width="400" height="250" alt="manual_nn_loss_curve" src="https://github.com/user-attachments/assets/9ec836ce-10c2-4b89-abfb-5cdb80c3fe15" /> | <img width="400" height="250" alt="ga_best_nn_loss_curve" src="https://github.com/user-attachments/assets/4c46d6c0-e125-48ca-b39b-eafa2d6b062d" /> |

Both models converge within 15 epochs with near-identical training and validation losses, confirming no overfitting.

### GA Progress

<img width="400" height="250" alt="ga_progress" src="https://github.com/user-attachments/assets/d0140612-464c-4e4b-80f9-f0d3f83d4212" />

---

## Demo (CLI Prediction)

The CLI takes raw vehicle specs as input, runs the full preprocessing pipeline internally, and returns a predicted MPG from the manual neural network.

Sample predictions from historical records:

| Specs | Origin | Actual MPG | Predicted | Error |
|---|---|---|---|---|
| 8cyl, 307disp, 130hp, 3504lb, 12acc, yr70 | USA | 18.0 | 16.29 | -1.71 |
| 6cyl, 250disp, 100hp, 3282lb, 15acc, yr71 | USA | 19.0 | 17.45 | -1.55 |
| 4cyl, 97disp, 52hp, 2130lb, 24.6acc, yr82 | Europe | 44.0 | 44.99 | +0.99 |
| 4cyl, 91disp, 67hp, 1965lb, 15.7acc, yr82 | Japan | 32.0 | 37.86 | +5.86 |

The model handles the 20-35 MPG range very well. Occasional overprediction for high-efficiency Japanese models from the early 1980s is likely due to limited samples in that specific cluster.

---

## Limitations

- Small dataset (398 records) limits both model complexity and GA search budget
- GA ran for only 5 generations with a population of 10, so a larger search would be more thorough
- The Power-to-Weight feature showed weaker linear correlation (r = -0.24) than anticipated, and its non-linear contribution was not formally isolated via ablation
- The CLI currently uses the manual NN as the inference engine (best R2); swapping to the GA model is straightforward
