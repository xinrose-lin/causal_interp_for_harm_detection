import matplotlib.pyplot as plt
import torch as t
import numpy as np
from scipy.stats import kurtosis


def training_plot(epoch_train_loss, epoch_test_acc):

    max_value = t.tensor(epoch_test_acc).topk(1).values.numpy().item()
    max_epoch = t.tensor(epoch_test_acc).topk(1).indices.numpy().item()

    plt.plot(epoch_train_loss, label="Train loss")
    plt.plot(epoch_test_acc, label="Test Accuracy")

    plt.text(max_epoch + 1, epoch_test_acc[-1] - 0.1, f"Max Test Accuracy: {max_value:.4f}, Epoch Index: {max_epoch}", 
            ha='left', va='center', fontsize=10, color="orange")

    # Add a vertical line at the max test accuracy epoch
    plt.axvline(x=max_epoch, color='orange', linestyle='--', label="Max Accuracy Epoch")

    plt.legend()
    plt.title("(Correlation) Localised Sparse Probe Training plot (dim=100)")

    # Add axis labels
    plt.xlabel("Epochs")
    plt.ylabel("Values")


## code for side by side boxplot
def boxplot(nonharmful_acts, harmful_acts): 
    ## mean for final token

    nonharmful_data = nonharmful_acts[:, -1, :].mean(0)  # mean along axes 0 and 1 for nonharmful_acts
    harmful_data = harmful_acts[:, -1, :].mean(0)      # mean along axes 0 and 1 for harmful_acts

    # Create a figure and axis
    plt.figure(figsize=(8, 6))

    # Plot nonharmful_acts boxplot at position 1 with a custom color
    box_nonharmful = plt.boxplot(nonharmful_data, positions=[1], patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))

    # Plot harmful_acts boxplot at position 2 with a custom color
    box_harmful = plt.boxplot(harmful_data, positions=[2], patch_artist=True, boxprops=dict(facecolor='lightcoral', color='red'))

    # Function to annotate max, min, and median values
    def annotate_boxplot(data, pos):
        # Extract the statistics from the boxplot
        median = t.median(data)
        minimum = t.min(data)
        maximum = t.max(data)

        plt.text(pos + 0.1, minimum, f"Min: {minimum:.2f}", ha='left', va='center', fontsize=10, color='blue')
        plt.text(pos + 0.1, median, f"Median: {median:.2f}", ha='left', va='center', fontsize=10, color='green')
        plt.text(pos + 0.1, maximum, f"Max: {maximum:.2f}", ha='left', va='center', fontsize=10, color='red')

    # Annotate both boxplots
    annotate_boxplot(nonharmful_data, 1)
    annotate_boxplot(harmful_data, 2)

    # Set the x-axis labels
    plt.xticks([1, 2], ['Non-Harmful Acts', 'Harmful Acts'])

    # Optional: Add labels and title
    plt.xlabel('Type of Acts')
    plt.ylabel('Mean Activation Value (n=64)')
    plt.title('Comparison of Non-Harmful vs Harmful last token Activations (dim=512)')

    # Display the plot
    plt.show()


def violin_plot(nonharmful,  harmful):
    
    data = [nonharmful, harmful]
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(9, 7))

    # Create the violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=True)

    # Function to annotate max, min, and median values
    def annotate_plot(data, pos):
        # Extract the statistics from the boxplot
        mean = t.mean(data)
        minimum = t.min(data)
        maximum = t.max(data)
        nonzero_counts = len(data)

        plt.text(pos + 0.1, minimum - 0.04, f"Min: {minimum:.2f}", ha='left', va='center', fontsize=10, color='blue')
        plt.text(pos + 0.1, mean + 0.1, f"Mean: {mean:.4f}", ha='left', va='center', fontsize=10, color='green')
        plt.text(pos + 0.1, maximum - 0.05, f"Max: {maximum:.2f}", ha='left', va='center', fontsize=10, color='red')
        # plt.text(pos - 0.1, maximum - 0.05, f"Non zero counts: {nonzero_counts}", ha='right', va='center', fontsize=10, color='black')

    # Annotate both boxplots
    annotate_plot(t.tensor(nonharmful), 1)
    annotate_plot(t.tensor(harmful), 2)

    # Customize appearance
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax.set_title("Distribution for Non-Zero Sparse Activation Values")
    ax.set_xlabel("Prompt type")
    ax.set_ylabel("Mean Activation Value (n=64)")
    ax.set_xticks([1, 2])  # Set x-tick positions
    ax.set_xticklabels([f"Non harmful prompts ", "Harmful prompts"])  # Set


## plot for up down comparison of distribution
def distribution_lineplot(harmful_acts, nonharmful_acts): 

    harmful_data = harmful_acts
    nonharmful_data = nonharmful_acts

    harmful_stats = {
        "max": harmful_data.max(),
        "mean": harmful_data.mean(),
        "median": harmful_data.median(),
        "min": harmful_data.min(), 
        "kurtosis": kurtosis(harmful_data)
    }
    nonharmful_stats = {
        "max": nonharmful_data.max(),
        "mean": nonharmful_data.mean(),
        "median": nonharmful_data.median(),
        "min": nonharmful_data.min(), 
        "kurtosis": kurtosis(nonharmful_data)
    }

    # Create two subplots stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot non-harmful data
    axes[0].plot(nonharmful_data, label="Non-harmful data", color='blue')
    axes[0].set_title("Averaged Non-Harmful Original Activations (n=64)")
    # axes[0].legend()
    axes[0].set_ylabel("Activation Value")

    # Annotate statistics
    axes[0].text(1.25, 0.9, f"Max: {nonharmful_stats['max']:.2f}", transform=axes[0].transAxes, fontsize=10, ha='right')
    axes[0].text(1.25, 0.8, f"Mean: {nonharmful_stats['mean']:.2f}", transform=axes[0].transAxes, fontsize=10, ha='right')
    axes[0].text(1.25, 0.7, f"Median: {nonharmful_stats['median']:.2f}", transform=axes[0].transAxes, fontsize=10, ha='right')
    axes[0].text(1.25, 0.6, f"Min: {nonharmful_stats['min']:.2f}", transform=axes[0].transAxes, fontsize=10, ha='right')
    axes[0].text(1.25, 0.5, f"Kurtosis: {nonharmful_stats['kurtosis']:.2f}", transform=axes[0].transAxes, fontsize=10, ha='right')

    # Plot harmful data
    axes[1].plot(harmful_data, label="Harmful data", color='red')
    axes[1].set_title("Averaged Harmful Original Activations (n=64)")

    axes[1].set_xlabel("Hidden dimensions Index")
    axes[1].set_ylabel("Activation Value")
    # Annotate statistics
    axes[1].text(1.25, 0.9, f"Max: {harmful_stats['max']:.2f}", transform=axes[1].transAxes, fontsize=10, ha='right')
    axes[1].text(1.25, 0.8, f"Mean: {harmful_stats['mean']:.2f}", transform=axes[1].transAxes, fontsize=10, ha='right')
    axes[1].text(1.25, 0.7, f"Median: {harmful_stats['median']:.2f}", transform=axes[1].transAxes, fontsize=10, ha='right')
    axes[1].text(1.25, 0.6, f"Min: {harmful_stats['min']:.2f}", transform=axes[1].transAxes, fontsize=10, ha='right')
    axes[1].text(1.25, 0.5, f"Kurtosis: {harmful_stats['kurtosis']:.2f}", transform=axes[1].transAxes, fontsize=10, ha='right')


    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
