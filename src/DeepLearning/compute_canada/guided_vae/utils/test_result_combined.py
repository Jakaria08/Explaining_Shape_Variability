import os

def find_extreme_rows(data_rows):
    filtered_data = [row for row in data_rows if float(row[9]) > 0.9 and float(row[11]) > 0.9]
    
    if filtered_data:
        def get_corr_sum(row):
            return float(row[9]) + float(row[11])
        
        def get_sap_sum(row):
            return float(row[10]) + float(row[12])
        
        sorted_corr_rows = sorted(filtered_data, key=get_corr_sum, reverse=True)[:5]
        sorted_sap_rows = sorted(data_rows, key=get_sap_sum, reverse=True)[:5]
        
        return sorted_corr_rows, sorted_sap_rows
    else:
        return [], []

# Read data from a text file
file_path = "C:\\Users\\Jakar\\Downloads\\Hippocampus_Study\\generate_synthetic_data\\contrastive_loss\\test.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Parse data into a list of rows
parsed_data = [line.strip().split("|") for line in lines]

# Find the extreme rows
corr_extreme_rows, sap_extreme_rows = find_extreme_rows(parsed_data)

if corr_extreme_rows:
    print("Top 5 Correlation Extreme Rows:")
    for idx, row in enumerate(corr_extreme_rows):
        print(f"Extreme Row {idx+1}: {row}")

if sap_extreme_rows:
    print("Top 5 SAP Extreme Rows:")
    for idx, row in enumerate(sap_extreme_rows):
        print(f"Extreme Row {idx+1}: {row}")

if not corr_extreme_rows and not sap_extreme_rows:
    print("No extreme rows found.")
