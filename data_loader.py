# src/data_loader.py

import pandas as pd
import numpy as np
import os

def load_real_data(filepath='data/solar_flare_real.csv'):
    """Load solar flare data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Real data CSV not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def generate_fake_data(rows=500):
    """Generate fake solar flare data for fallback or testing."""
    np.random.seed(42)
    data = {
        "Sunspot Number": np.random.randint(0, 500, rows),
        "Radio Flux": np.random.randint(50, 250, rows),
        "X-Ray Emission": np.random.rand(rows)*10,
        "Flare": np.random.randint(0, 2, rows),
    }
    df = pd.DataFrame(data)
    return df

def load_data(real_file='data/solar_flare_real.csv'):
    """General wrapper to return real or fake data depending on availability."""
    try:
        df = load_real_data(real_file)
        print("✅ Loaded real solar flare data.")
    except FileNotFoundError:
        df = generate_fake_data()
        print("⚡ File not found. Loading fake data.")
    return df
