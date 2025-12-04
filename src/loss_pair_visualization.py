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

args = parser.parse_args()
if args.isbsub:
    postifx = "_bsub"
else:
    postifx = ""

# Function to plot training vs validation loss pairs
def plot_loss_pairs(loss_pairs, output_image):
    train_losses = [pair[0] for pair in loss_pairs]
    val_losses = [pair[1] for pair in loss_pairs]

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image)
    plt.close()

# Main execution
if __name__ == "__main__":
    # Open the JSON file and read loss pairs
    with open("loss_pairs"+postifx+".json", "r") as f:
        loss_pairs = [(entry["train_loss"], entry["val_loss"]) for entry in json.load(f)]
    
    # Plot and save the loss pairs
    plot_loss_pairs(loss_pairs, output_image="loss_pair_plot"+postifx+".png")