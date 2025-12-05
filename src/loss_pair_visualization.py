"""
This script reads training and validation loss pairs from a JSON file and
plots them using matplotlib. The resulting plot is saved as a PNG image.
"""

from matplotlib import pyplot as plt
import json
import argparse

# Argument parser for bsub flag (for jobs on cluster)
parser = argparse.ArgumentParser()
parser.add_argument("--isbsub", action="store_true")

# NEW: start/end index selection
parser.add_argument("--start", type=int, default=0, help="Start index of epochs to plot")
parser.add_argument("--end", type=int, default=None, help="End index of epochs to plot")

args = parser.parse_args()

postifx = "_bsub" if args.isbsub else ""

# Function to plot training vs validation loss pairs
def plot_loss_pairs(loss_pairs, output_image, start_idx=0, end_idx=None):

    # Slice the data
    sliced = loss_pairs[start_idx:end_idx]

    train_losses = [pair[0] for pair in sliced]
    val_losses = [pair[1] for pair in sliced]

    # Generate x values based on the sliced range
    x_values = list(range(start_idx, start_idx + len(sliced)))

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, train_losses, label='Training Loss', marker='o')
    plt.plot(x_values, val_losses, label='Validation Loss', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss (epochs {start_idx}:{end_idx})')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image)
    plt.close()


# Main execution
if __name__ == "__main__":
    # Open the JSON file and read loss pairs
    with open("loss_pairs" + postifx + ".json", "r") as f:
        loss_pairs = [(entry["train_loss"], entry["val_loss"]) for entry in json.load(f)]
    
    # Plot and save the loss pairs
    plot_loss_pairs(
        loss_pairs,
        output_image="loss_pair_plot" + postifx + ".png",
        start_idx=args.start,
        end_idx=args.end
    )
