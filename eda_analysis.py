import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Importing the pipeline from your teammate's file
from data_prep import prepare_data

def run_visual_audit():
    """
    Role: Create visual audit of unscaled data to identify trends 
    and justify engineered features like Power_to_Weight.
    """
    # 1. Get the data from the pipeline[cite: 1]
    # We only need 'full_df' (the 6th item returned) for human-readable EDA[cite: 1]
    _, _, _, _, _, full_df = prepare_data()

    print("Generating Visualizations...")

    # 2. Set the professional aesthetic style
    sns.set_theme(style="whitegrid")
    
    # Create a 'results' folder if it doesn't exist to keep the repo clean
    if not os.path.exists('results'):
        os.makedirs('results')

    # 3. Pairplot: Visualizing distributions and relationships[cite: 1]
    cols_to_plot = ['MPG', 'Horsepower', 'Weight', 'Acceleration', 'Power_to_Weight']
    
    pair_plot = sns.pairplot(full_df[cols_to_plot], diag_kind='kde', corner=True)
    pair_plot.fig.suptitle("Vehicle Data: Pairwise Relationships & Distributions", y=1.02)
    
    # Save the plot as an image for the team
    pair_plot.savefig('results/pairplot_audit.png')
    print("Saved: results/pairplot_audit.png")
    plt.show()

    # 4. Correlation Heatmap: Statistical verification[cite: 1]
    plt.figure(figsize=(10, 8))
    correlation_matrix = full_df[cols_to_plot].corr()
    
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix (Unscaled Data)")
    
    # Save the heatmap as an image for the team
    plt.savefig('results/correlation_heatmap.png')
    print("Saved: results/correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    run_visual_audit()