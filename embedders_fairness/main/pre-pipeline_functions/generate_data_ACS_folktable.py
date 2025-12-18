import pandas as pd
from folktables import ACSDataSource, ACSPublicCoverage, ACSMobility


def export_acs_dataset(dataset_class, dataset_name, n_samples=100_000, out_dir="./Data"):
    # Load ACS for year 2018, 1-year horizon, person survey
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=None, download=True)

    # Transform into features / labels / groups
    features, labels, _ = dataset_class.df_to_numpy(acs_data)

    # Build DataFrame
    X = pd.DataFrame(features, columns=dataset_class.features)
    y = pd.Series(labels, name="target")

    # Merge
    df = pd.concat([X, y], axis=1)

    # Display size before sampling
    print(f"{dataset_name}: initial size = {df.shape}")

    # Sample n_samples rows
    sampled = df.sample(n=n_samples, random_state=42)

    # 1. Normal dataset with target
    df_with_target = sampled.copy()
    df_with_target.to_csv(f"{out_dir}/{dataset_name}_with_target.csv", index=False)
    print(f"CSV generated: {dataset_name}_with_target.csv")

    # 2. Normal dataset without target
    df_without_target = sampled.drop(columns=["target"]).copy()
    df_without_target.to_csv(f"{out_dir}/{dataset_name}_without_target.csv", index=False)
    print(f"CSV generated: {dataset_name}_without_target.csv")

    # # 3. One-hot dataset with target
    # df_one_hot = pd.get_dummies(df_with_target.drop(columns=["target"]), drop_first=True)
    # df_one_hot["target"] = df_with_target["target"]
    # df_one_hot.to_csv(f"{out_dir}/{dataset_name}_one_hot_with_target.csv", index=False)
    # print(f"CSV generated: {dataset_name}_one_hot_with_target.csv")
    # print("-" * 50)


# === Export Public Coverage ===
export_acs_dataset(ACSPublicCoverage, "ACSPublicCoverage")

# === Export Mobility ===
#export_acs_dataset(ACSMobility, "ACSMobility")