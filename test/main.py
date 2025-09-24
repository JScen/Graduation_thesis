# main.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data(file_path: str = "excel.xlsx", sheet_name: str = "Sheet1") -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name=sheet_name)
