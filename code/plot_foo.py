import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
from matplotlib import rcParams

def plot_metric(run_name, metric="accuracy"):
    """
    Plots the specified metric (accuracy or AUC) for training and validation sets and saves the figure as a PDF,
    formatted for inclusion in a LaTeX document.

    Args:
        run_name (str): Name of the run directory containing the data files.
        metric (str): The metric to plot ("accuracy" or "AUC"). Defaults to "accuracy".
    """
    # Define base and data paths
    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_PATH.joinpath("data").joinpath(run_name)

    # File paths for training and validation data
    training_file = DATA_PATH.joinpath(f"train_{metric}.csv")
    validation_file = DATA_PATH.joinpath(f"validation_{metric}.csv")

    # Read data using Polars without headers, treating the first row as data
    training_data = pl.read_csv(training_file, has_header=False)
    validation_data = pl.read_csv(validation_file, has_header=False)

    # Assign meaningful column names
    training_data.columns = ["epoch", "ignore", metric]
    validation_data.columns = ["epoch", "ignore", metric]   

    # Convert Polars DataFrames to Pandas for compatibility with Seaborn
    training_df = training_data.to_pandas()
    validation_df = validation_data.to_pandas()

    # Add a 'set' column to differentiate between training and validation
    training_df["set"] = "Training"
    validation_df["set"] = "Validation"

    # Combine data into a single DataFrame
    combined_df = pl.DataFrame(training_df).extend(pl.DataFrame(validation_df)).to_pandas()

    # Configure LaTeX-style fonts
    rcParams["font.family"] = "serif"
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 10

    # Plot the specified metric
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))  # Resize for a typical LaTeX document
    sns.lineplot(data=combined_df, x="epoch", y=metric, hue="set", marker="o")

    # Add plot labels and title
    plt.title(f"{metric.capitalize()} vs Epochs for {run_name}", fontsize=22)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel(metric.capitalize(), fontsize=18)
    plt.legend(title="Dataset", fontsize=12)
    plt.tight_layout()

    # Save plot to PDF in the data directory
    output_file = DATA_PATH.joinpath(f"{metric}_plot.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")

    # Show the plot
    plt.show()

    print(f"Plot saved to {output_file}")

plot_metric("run_18","auc")