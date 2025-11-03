from src.data_loader.dataloader import DataLoader


def main():
    data = DataLoader(gender="female")
    print(f"\n=== Loaded {data.gender.upper()} dataset ===")
    print(f"Counties: {len(data.get_all_counties())}\n")

    # Print ALL variables (name + a tiny preview)
    for var_name, series in data.get_variables().items():
        # Get the first 3 items for a compact preview
        items = list(series.items())
        preview = ", ".join(f"{k}: {v}" for k, v in items[:3])
        print(f"{var_name}: [{len(series)} counties]  ->  {preview}")


if __name__ == "__main__":
    main()
