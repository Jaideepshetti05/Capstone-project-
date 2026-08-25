import pandas as pd
import csv

print("Testing Python engine...")
try:
    df_py = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv', engine='python')
    print(f"Python engine success: shape={df_py.shape}")
except Exception as e:
    print(f"Python engine failed: {e}")

print("\nTesting line by line with csv.reader...")
with open('feature_vectors_syscallsbinders_frequency_5_Cat.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    header = next(reader)
    expected_len = len(header)
    print(f"Header columns: {expected_len}")
    
    bad_rows = []
    total_rows = 0
    for idx, row in enumerate(reader, start=1):
        total_rows += 1
        if len(row) != expected_len:
            bad_rows.append((idx, len(row)))
            if len(bad_rows) <= 5:
                print(f"Row {idx} has {len(row)} items instead of {expected_len}")
        if total_rows % 5000 == 0:
            print(f"Processed {total_rows} rows...")

print(f"Total data rows read: {total_rows}")
print(f"Total bad rows: {len(bad_rows)}")

# Also test pyarrow engine if available
try:
    df_arrow = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv', engine='pyarrow')
    print(f"Pyarrow engine success: shape={df_arrow.shape}")
except Exception as e:
    print(f"Pyarrow engine: {e}")
