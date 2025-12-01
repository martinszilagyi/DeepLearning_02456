import os

def count_parquet_files(root_dir):
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        count += sum(1 for f in filenames if f.endswith('.parquet'))
    return count

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "csvs", "parquets")
    total_files = count_parquet_files(folder_path)
    print(f"Total parquet files found: {total_files}")