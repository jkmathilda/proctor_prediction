"""
Split particle-size data into soil-type fractions using the DIN grain-size
boundaries shown in the Kornverteilung graph:

    Clay (Clay)     : d < 0.002 mm
    Silt (Silt) : 0.002 <= d < 0.063 mm
    Sand           : 0.063 <= d < 2 mm
    Gravel (Gravel)  : d >= 2 mm

The dataset stores cumulative mass-% passing at those three boundary diameters,
so each fraction is obtained by differencing the cumulative curve.
"""

import argparse
import pandas as pd

# Cumulative-passing columns at the DIN boundary diameters
P002 = "psd_passing_at_0_002mm_pct"   # % finer than 0.002 mm
P063 = "psd_passing_at_0_063mm_pct"   # % finer than 0.063 mm
P2   = "psd_passing_at_2mm_pct"       # % finer than 2 mm


def split_soil_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add clay/silt/sand/gravel mass-% columns and a dominant soil type."""
    out = df.copy()

    # Fractions from the cumulative curve (clamp tiny negatives from rounding)
    out["clay_pct"]     = out[P002].clip(lower=0)
    out["silt_pct"] = (out[P063] - out[P002]).clip(lower=0)
    out["sand_pct"]         = (out[P2]   - out[P063]).clip(lower=0)
    out["gravel_pct"]  = (100 - out[P2]).clip(lower=0)

    fraction_cols = ["clay_pct", "silt_pct", "sand_pct", "gravel_pct"]

    # Dominant (largest) fraction per sample
    label = {
        "clay_pct": "Clay",
        "silt_pct": "Silt",
        "sand_pct": "Sand",
        "gravel_pct": "Gravel",
    }
    out["dominant_soil_type"] = out[fraction_cols].idxmax(axis=1).map(label)

    return out


def main():
    ap = argparse.ArgumentParser(description="Split PSD data into DIN soil types.")
    ap.add_argument("input", nargs="?", default="train.csv", help="input CSV")
    ap.add_argument("-o", "--output", default="train_soil_types.csv", help="output CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    result = split_soil_types(df)
    result.to_csv(args.output, index=False)
    sand = result[result['dominant_soil_type'] == "Sand"]
    sand.to_csv("sand.csv", index = False)

    gravel = result[result['dominant_soil_type'] == "Gravel"]
    gravel.to_csv("gravel.csv", index = False)
    silt = result[result['dominant_soil_type'] == "Silt"]
    silt.to_csv("silt.csv", index = False)


    print(f"Wrote {len(result)} rows -> {args.output}")
    print("\nDominant soil type counts:")
    print(result["dominant_soil_type"].value_counts().to_string())
    print("\nMean fractions (%):")
    print(result[["clay_pct", "silt_pct",
                  "sand_pct", "gravel_pct"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()