# 02456 Deep Learning Project: Increasing Maritime Domain Awareness Using Spatio-Temporal Sequential Data

**Project goal:**  
Use Automatic Identification System (AIS) data from vessels to predict their trajectories for the next hour in the waters around Denmark.

---

## Data

Raw data is available from:  
[https://aisdata.ais.dk/2023/aisdk-2023-02.zip](https://aisdata.ais.dk/2023/aisdk-2023-02.zip)

---

## Data Preprocessing

1. Extract the CSV files inside the zipped folder to the `./csvs` directory.

2. From the `src` folder, run the following scripts in order:

    1. **`ais_to_parquet.py`**  
       Converts raw CSV data to parquet files using different techniques.

    2. **`normalize_parquet_data.py`**  
       Extends the raw parquet columns with new useful features and normalizes all columns globally using Z-score normalization.  
       This script also generates a JSON file containing normalization parameters (means and standard deviations).

    3. **`final_clean.py`**  
       Organizes the parquet files into separate folders and extends columns with enumerated non-numeric types.

> **Note:** The responsibilities of these scripts may seem arbitrary, but they are designed this way because the data processing steps are very time-consuming.

---

## Training

1. Run `main_baseline.py` and  `main_improvement.py` 
   This scripts handle all neural network training components. They do not accept tunable parameters externally.  
   After training, they save the model to a `.pth` file, applies the model to the test set, and generates a `.json` file with all predicted data.  
   (For debugging purposes, an additional JSON file is created to visualize training predictions.)

2. **Optional:** Visualization on a map  
   Create a visualization based on two JSON files:  
   - One from the normalization step (for denormalization)  
   - One containing the test predictions (for visualizing predicted points)  

   This outputs an `.html` file showing:  
   - Known path (blue)  
   - Ground truth (green)  
   - Predictions (red)

3. **Optional:** Plot training and validation loss  
   Visualize loss over a specified epoch range (if no range is given, the whole range is used).  
   This saves a `.png` plot in the root folder.  

   Example usage:  
   ```bash
   python ./loss_pair_visualization.py --start 0 --end 150

---

## Other files

Additional scripts and the data are also available in the repository. These scripts served as backups at each milestone.
Results can be found under its corresponding folder.
