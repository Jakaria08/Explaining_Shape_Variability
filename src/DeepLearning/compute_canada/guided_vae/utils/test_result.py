import os

def find_extreme_rows(data_rows):
    max_corr_row = max(data_rows, key=lambda row: float(row[9]))
    max_sap_row = max(data_rows, key=lambda row: float(row[10]))
    max_corr2_row = max(data_rows, key=lambda row: float(row[11]))
    max_sap2_row = max(data_rows, key=lambda row: float(row[12]))
    min_euclidean_row = min(data_rows, key=lambda row: float(row[13]))
    
    return max_corr_row, max_sap_row, max_corr2_row, max_sap2_row, min_euclidean_row

# Read data from a text file
file_path = "C:\\Users\\Jakar\\Downloads\\Hippocampus_Study\\generate_synthetic_data\\correlation_decorrelation_loss\\test.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Parse data into a list of rows
parsed_data = [line.strip().split("|") for line in lines]

# Find the extreme rows
extreme_rows = find_extreme_rows(parsed_data)

for idx, row in enumerate(extreme_rows):
    print(f"Extreme Row {idx+1}: {row}")
