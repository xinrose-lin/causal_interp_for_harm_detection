import matplotlib.pyplot as plt
import torch as t


def training_plot(epoch_train_loss, epoch_test_acc):
    plt.plot(epoch_train_loss, label="Train loss")
    plt.plot(epoch_test_acc, label="Test Accuracy")

    # # Annotate the final values
    # plt.text(len(epoch_train_loss) - 1, epoch_train_loss[-1], f"{epoch_train_loss[-1]:.2f}", 
    #         ha='left', va='center', fontsize=10, color="blue")
    # plt.text(len(epoch_test_acc) - 1, epoch_test_acc[-1], f"{epoch_test_acc[-1]:.2f}", 
    #         ha='left', va='center', fontsize=10, color="orange")

    plt.legend()
    plt.title("(dim=512) acts probe train loss and test acc")

    # Add axis labels
    plt.xlabel("Epochs")
    plt.ylabel("Values")

    # print(epoch_test_acc)

## EDA: do we observe different 
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


# import matplotlib.pyplot as plt
# import numpy as np

def violin_plot():
    
    data = [nonzero_nonharmful, nonzero_harmful]
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
    annotate_plot(t.tensor(nonzero_nonharmful), 1)
    annotate_plot(t.tensor(nonzero_harmful), 2)

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

