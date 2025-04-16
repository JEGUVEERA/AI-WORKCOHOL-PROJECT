# utils/file_loader.py
import pandas as pd
import os

# ✅ Load and clean uploaded file
def load_file(file):
    ext = os.path.splitext(file.name)[-1]
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(file, encoding="ISO-8859-1")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
    else:
        return None

    # Convert numeric values from strings with symbols
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0], errors="coerce")

    return df
