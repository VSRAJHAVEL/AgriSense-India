"""
Generate Crop Recommendation Dataset for Indian crops.
Creates a CSV with N, P, K, temperature, humidity, pH, rainfall, and crop label.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Indian crop parameters: (N, P, K, temp, humidity, pH, rainfall)
CROP_PARAMS = {
    "rice":       {"N": (80, 10), "P": (40, 8), "K": (40, 8), "temp": (28, 3), "hum": (70, 8), "ph": (6.2, 0.4), "rain": (1200, 150)},
    "wheat":      {"N": (100,12), "P": (50, 8), "K": (35, 6), "temp": (18, 3), "hum": (50, 8), "ph": (6.8, 0.4), "rain": (500, 80)},
    "maize":      {"N": (40, 8),  "P": (60, 10),"K": (20, 5), "temp": (28, 3), "hum": (62, 8), "ph": (6.5, 0.5), "rain": (750, 100)},
    "cotton":     {"N": (50, 10), "P": (50, 8), "K": (50, 8), "temp": (30, 3), "hum": (52, 8), "ph": (7.0, 0.5), "rain": (800, 120)},
    "groundnut":  {"N": (20, 5),  "P": (60, 8), "K": (40, 8), "temp": (28, 3), "hum": (60, 7), "ph": (6.2, 0.4), "rain": (600, 80)},
    "sugarcane":  {"N": (60, 10), "P": (40, 8), "K": (40, 8), "temp": (30, 4), "hum": (68, 8), "ph": (7.0, 0.5), "rain": (1300, 150)},
    "banana":     {"N": (60, 10), "P": (50, 8), "K": (70, 10),"temp": (29, 3), "hum": (72, 8), "ph": (6.8, 0.4), "rain": (1250, 150)},
    "coconut":    {"N": (50, 8),  "P": (40, 8), "K": (80, 10),"temp": (27, 3), "hum": (75, 8), "ph": (6.5, 0.5), "rain": (1500, 200)},
    "millets":    {"N": (30, 6),  "P": (30, 6), "K": (20, 5), "temp": (29, 3), "hum": (52, 8), "ph": (6.2, 0.6), "rain": (550, 100)},
    "blackgram":  {"N": (20, 5),  "P": (60, 8), "K": (20, 5), "temp": (30, 3), "hum": (60, 7), "ph": (6.8, 0.4), "rain": (500, 80)},
    "greengram":  {"N": (20, 5),  "P": (50, 8), "K": (20, 5), "temp": (30, 3), "hum": (60, 7), "ph": (6.8, 0.4), "rain": (475, 80)},
    "redgram":    {"N": (20, 5),  "P": (60, 8), "K": (20, 5), "temp": (27, 3), "hum": (60, 7), "ph": (6.2, 0.4), "rain": (700, 80)},
    "turmeric":   {"N": (60, 10), "P": (30, 6), "K": (60, 10),"temp": (27, 3), "hum": (70, 8), "ph": (6.5, 0.5), "rain": (1100, 150)},
    "sesame":     {"N": (35, 6),  "P": (25, 5), "K": (25, 5), "temp": (31, 3), "hum": (50, 7), "ph": (6.8, 0.6), "rain": (525, 80)},
    "sunflower":  {"N": (40, 8),  "P": (30, 6), "K": (20, 5), "temp": (27, 3), "hum": (55, 7), "ph": (6.8, 0.4), "rain": (625, 80)},
    "chilli":     {"N": (50, 8),  "P": (30, 6), "K": (30, 6), "temp": (27, 3), "hum": (60, 7), "ph": (6.8, 0.4), "rain": (800, 120)},
    "lentil":     {"N": (15, 4),  "P": (40, 6), "K": (18, 4), "temp": (20, 3), "hum": (48, 7), "ph": (6.5, 0.4), "rain": (380, 60)},
    "mustard":    {"N": (50, 8),  "P": (30, 5), "K": (18, 4), "temp": (18, 3), "hum": (48, 7), "ph": (6.5, 0.4), "rain": (320, 50)},
}

SAMPLES_PER_CROP = 150

def generate():
    rows = []
    for crop, params in CROP_PARAMS.items():
        for _ in range(SAMPLES_PER_CROP):
            row = {
                "N": max(0, int(np.random.normal(*params["N"]))),
                "P": max(0, int(np.random.normal(*params["P"]))),
                "K": max(0, int(np.random.normal(*params["K"]))),
                "temperature": round(np.random.normal(*params["temp"]), 2),
                "humidity": round(np.clip(np.random.normal(*params["hum"]), 20, 99), 2),
                "ph": round(np.clip(np.random.normal(*params["ph"]), 3.5, 9.5), 2),
                "rainfall": round(max(50, np.random.normal(*params["rain"])), 2),
                "label": crop
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/Crop_recommendation.csv", index=False)
    print(f"Generated {len(df)} samples for {len(CROP_PARAMS)} crops")
    print(f"Saved to data/Crop_recommendation.csv")
    print(f"\nCrop distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    generate()
