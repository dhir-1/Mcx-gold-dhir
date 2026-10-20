"""
preprocess.py - Preprocess MCX Gold Futures Bhavcopy Data

Step 1 of the Commodity Derivatives Intelligence project.
This script cleans raw daily MCX bhavcopy CSV files for gold futures contracts.

Usage:
    python preprocess.py data out
"""

import sys
import os
from pathlib import Path
import pandas as pd


def main():
    # Read input and output folder paths from command line arguments
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "out"

    # Create the output directory if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    input_path = Path(input_folder)
    csv_files = sorted(list(input_path.glob("*.csv")))
    if not csv_files:
        print(f"Error: No CSV files found in '{input_folder}'")
        return

    # Track data quality issues found during validation checks
    issues = []

    # Define quote unit in grams for converting prices to per-gram
    quote_unit_grams = {
        "GOLDM": 10,
        "GOLDTEN": 10,
        "GOLDGUINEA": 8,
        "GOLDPETAL": 1
    }

    # Target instrument and symbols to keep
    target_instrument = "FUTCOM"
    allowed_symbols = {"GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"}

    processed_dfs = []

    # -------------------------------------------------------------
    # Step 1: Read every CSV in the input folder
    # -------------------------------------------------------------
    for file_path in csv_files:
        filename = file_path.name
        file_stem = file_path.stem

        # Read the raw CSV file
        df_raw = pd.read_csv(file_path)

        # Strip leading and trailing whitespace from column names
        df_raw.columns = df_raw.columns.str.strip()

        # Strip whitespace from key text columns
        if "Symbol" in df_raw.columns:
            df_raw["Symbol"] = df_raw["Symbol"].astype(str).str.strip()
        if "Instrument Name" in df_raw.columns:
            df_raw["Instrument Name"] = df_raw["Instrument Name"].astype(str).str.strip()
        if "Expiry Date" in df_raw.columns:
            df_raw["Expiry Date"] = df_raw["Expiry Date"].astype(str).str.strip()

        # Check: Verify if filename matches the symbol and expiry inside the file
        unique_contracts = df_raw[["Symbol", "Expiry Date"]].drop_duplicates()
        for _, row in unique_contracts.iterrows():
            sym = row["Symbol"]
            exp = row["Expiry Date"]
            expected_stem = f"{sym}_{exp}".upper()
            if file_stem.upper() != expected_stem:
                issues.append({
                    "file": filename,
                    "problem": "File name mismatch",
                    "detail": f"Filename '{filename}' does not match content Symbol='{sym}', Expiry='{exp}'"
                })

        # -------------------------------------------------------------
        # Step 2: Keep only rows where Instrument Name is FUTCOM and Symbol
        # is GOLDM, GOLDTEN, GOLDGUINEA or GOLDPETAL
        # -------------------------------------------------------------
        mask = (df_raw["Instrument Name"] == target_instrument) & (df_raw["Symbol"].isin(allowed_symbols))
        df_filtered = df_raw[mask].copy()

        if df_filtered.empty:
            continue

        # Keep source file name for logging issues
        df_filtered["_source_file"] = filename

        # -------------------------------------------------------------
        # Step 3: Drop Instrument Name, Option Type, Strike Price, Volume(In 000's)
        # -------------------------------------------------------------
        cols_to_drop = ["Instrument Name", "Option Type", "Strike Price", "Volume(In 000's)"]
        df_filtered = df_filtered.drop(columns=[c for c in cols_to_drop if c in df_filtered.columns])

        # -------------------------------------------------------------
        # Step 4: Parse Date and Expiry Date into real dates
        # -------------------------------------------------------------
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], format="%d %b %Y", errors="coerce")
        df_filtered["Expiry Date"] = pd.to_datetime(df_filtered["Expiry Date"], format="%d%b%Y", errors="coerce")

        # -------------------------------------------------------------
        # Step 5: Rename columns to simple lowercase names
        # -------------------------------------------------------------
        rename_map = {
            "Date": "date",
            "Symbol": "symbol",
            "Expiry Date": "expiry_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Previous Close": "prev_close",
            "Volume(Lots)": "volume_lots",
            "Value(Lacs)": "value_lacs",
            "Open Interest(Lots)": "open_interest_lots"
        }
        df_filtered = df_filtered.rename(columns=rename_map)

        # Ensure numeric fields are stored as proper numeric types
        numeric_cols = ["open", "high", "low", "close", "prev_close", "volume_lots", "value_lacs", "open_interest_lots"]
        for c in numeric_cols:
            if c in df_filtered.columns:
                df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")

        processed_dfs.append(df_filtered)

    if not processed_dfs:
        print("No valid rows found after preprocessing.")
        return

    # Combine all individual contract DataFrames into one table
    df = pd.concat(processed_dfs, ignore_index=True)

    # -------------------------------------------------------------
    # Step 6: Add close_per_gram, traded, and days_to_expiry columns
    # -------------------------------------------------------------
    # 6a: close_per_gram = close / quote_unit_grams
    df["close_per_gram"] = df["close"] / df["symbol"].map(quote_unit_grams)

    # 6b: traded = True if volume_lots > 0, else False
    df["traded"] = df["volume_lots"] > 0

    # 6c: days_to_expiry = expiry_date minus date, in calendar days
    df["days_to_expiry"] = (df["expiry_date"] - df["date"]).dt.days

    # -------------------------------------------------------------
    # CHECKS: Run validation checks across the combined dataset
    # -------------------------------------------------------------

    # Check: Missing weekdays between the first and last date of each (symbol, expiry_date)
    for (sym, exp), group in df.groupby(["symbol", "expiry_date"]):
        min_date = group["date"].min()
        max_date = group["date"].max()
        src_file = group["_source_file"].iloc[0]
        all_weekdays = pd.bdate_range(start=min_date, end=max_date)
        present_dates = set(group["date"])
        missing_days = sorted(set(all_weekdays) - present_dates)
        for missing_day in missing_days:
            issues.append({
                "file": src_file,
                "problem": "Missing weekday",
                "detail": f"{sym} ({exp.strftime('%Y-%m-%d')}): {missing_day.strftime('%Y-%m-%d')} ({missing_day.strftime('%A')}) missing"
            })

    # Check: Duplicate (date, symbol, expiry_date) rows
    dup_mask = df.duplicated(subset=["date", "symbol", "expiry_date"], keep=False)
    for _, row in df[dup_mask].iterrows():
        issues.append({
            "file": row["_source_file"],
            "problem": "Duplicate (date, symbol, expiry_date)",
            "detail": f"{row['symbol']} on {row['date'].strftime('%Y-%m-%d')}, expiry {row['expiry_date'].strftime('%Y-%m-%d')}"
        })

    # Check: Close missing or <= 0 on a traded day
    invalid_close_mask = df["traded"] & (df["close"].isna() | (df["close"] <= 0))
    for _, row in df[invalid_close_mask].iterrows():
        issues.append({
            "file": row["_source_file"],
            "problem": "Invalid close on traded day",
            "detail": f"{row['symbol']} on {row['date'].strftime('%Y-%m-%d')}: Close={row['close']}, Volume={row['volume_lots']}"
        })

    # Check: High < Low
    high_low_mask = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
    for _, row in df[high_low_mask].iterrows():
        issues.append({
            "file": row["_source_file"],
            "problem": "High < Low",
            "detail": f"{row['symbol']} on {row['date'].strftime('%Y-%m-%d')}: High={row['high']}, Low={row['low']}"
        })

    # Save all identified issues to out/issues_log.csv
    issues_df = pd.DataFrame(issues, columns=["file", "problem", "detail"])
    issues_log_path = Path(output_folder) / "issues_log.csv"
    issues_df.to_csv(issues_log_path, index=False)

    # -------------------------------------------------------------
    # Step 7: Sort by symbol, expiry_date, date and remove exact duplicate rows
    # -------------------------------------------------------------
    # Drop temporary source file tracking column before saving
    df_clean = df.drop(columns=["_source_file"])
    df_clean = df_clean.sort_values(by=["symbol", "expiry_date", "date"]).drop_duplicates().reset_index(drop=True)

    # Format dates as YYYY-MM-DD strings for consistent CSV output
    df_clean_output = df_clean.copy()
    df_clean_output["date"] = df_clean_output["date"].dt.strftime("%Y-%m-%d")
    df_clean_output["expiry_date"] = df_clean_output["expiry_date"].dt.strftime("%Y-%m-%d")

    # -------------------------------------------------------------
    # Step 8: Save out/gold_futures_clean.csv
    # -------------------------------------------------------------
    clean_csv_path = Path(output_folder) / "gold_futures_clean.csv"
    df_clean_output.to_csv(clean_csv_path, index=False)

    # -------------------------------------------------------------
    # PRINT AT THE END: Summary of processed data and validation checks
    # -------------------------------------------------------------
    print("=" * 70)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 70)

    # Print total rows and overall date range
    first_overall_date = df_clean["date"].min().strftime("%Y-%m-%d")
    last_overall_date = df_clean["date"].max().strftime("%Y-%m-%d")
    unique_dates_count = df_clean["date"].nunique()

    print(f"Total clean rows: {len(df_clean)}")
    print(f"Date range: {first_overall_date} to {last_overall_date} ({unique_dates_count} trading days)")
    print()

    # Print contract summary table
    summary_records = []
    for (sym, exp), group in df_clean.groupby(["symbol", "expiry_date"]):
        summary_records.append({
            "symbol": sym,
            "expiry_date": exp.strftime("%Y-%m-%d"),
            "first_date": group["date"].min().strftime("%Y-%m-%d"),
            "last_date": group["date"].max().strftime("%Y-%m-%d"),
            "number_of_days": len(group),
            "traded_days": int(group["traded"].sum()),
            "avg_volume_lots": round(group["volume_lots"].mean(), 2)
        })

    summary_df = pd.DataFrame(summary_records)
    print("Contract Summary Table:")
    print(summary_df.to_string(index=False))
    print()

    # Print summary of issues found
    print(f"Number of issues found: {len(issues)}")
    if len(issues) > 0:
        counts = issues_df["problem"].value_counts()
        for problem_name, count in counts.items():
            print(f"  - {problem_name}: {count}")
    else:
        print("  - Checks passed: No filename mismatches, missing weekdays, duplicates, or price anomalies.")
    print(f"Issues log saved to: {issues_log_path}")
    print(f"Clean data saved to: {clean_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
