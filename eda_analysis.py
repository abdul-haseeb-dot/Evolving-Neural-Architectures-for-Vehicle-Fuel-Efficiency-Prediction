import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from data_prep import prepare_data

def run_visual_audit():
    """
    Role: Create visual audit of unscaled data.
    Fix: Handles One-Hot Encoded columns (USA, Europe, Japan) for Boxplots.
    """
    # 1. Get the data from your pipeline
    _, _, _, _, _, full_df = prepare_data()

    print("Generating Visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Create a 'results' folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # 2. Pairplot & Heatmap
    # Using 'Power_to_Weight' (matching your excel screenshot)
    cols_to_plot = ['MPG', 'Horsepower', 'Weight', 'Acceleration', 'Power_to_Weight']
    
    # Filter only columns that exist to avoid errors
    existing_cols = [c for c in cols_to_plot if c in full_df.columns]
    
    pair_plot = sns.pairplot(full_df[existing_cols], diag_kind='kde', corner=True)
    pair_plot.savefig('results/pairplot_audit.png')
    print("Saved: results/pairplot_audit.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(full_df[existing_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig('results/correlation_heatmap.png')
    print("Saved: results/correlation_heatmap.png")

    # 3. BOXPLOT: Reconstruct 'Origin' from One-Hot columns
    # We check which column is 'TRUE' for each row to create a label
    def identify_origin(row):
        if row['USA'] == True: return 'USA'
        if row['Japan'] == True: return 'Japan'
        if row['Europe'] == True: return 'Europe'
        return 'Other'

    # Create a temporary column for plotting
    full_df['Origin_Group'] = full_df.apply(identify_origin, axis=1)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Origin_Group', y='MPG', data=full_df, palette='Set2')
    plt.title("Fuel Efficiency (MPG) by Region of Origin")
    plt.xlabel("Region")
    plt.ylabel("Miles Per Gallon (MPG)")
    
    plt.savefig('results/origin_boxplot.png')
    print("Saved: results/origin_boxplot.png")
    
    plt.show()

if __name__ == "__main__":
    run_visual_audit()
