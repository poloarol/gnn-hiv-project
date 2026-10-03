import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

def sampler(data):
    # Count classes
    neg_class = (data["HIV_active"] == 0).sum()
    pos_class = (data["HIV_active"] == 1).sum()

    print(f"Original negatives: {neg_class}")
    print(f"Original positives: {pos_class}")

    # Select positive samples
    positive_data = data[data["HIV_active"] == 1]

    # Randomly sample positive molecules with replacement
    additional_pos = positive_data.sample(
        n=neg_class - pos_class,
        replace=True,
        random_state=42
    )

    # Combine original + oversampled positives
    data = pd.concat(
        [data, additional_pos],
        ignore_index=True
    )

    # Shuffle the dataset
    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Create a new row ID
    data["index"] = data.index

    print("\nAfter oversampling:")
    print(data["HIV_active"].value_counts())

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/raw/HIV.csv"
    )

    parser.add_argument(
        "--output_train_csv",
        type=str,
        default="data/raw/HIV_train.csv"
    )
    
    parser.add_argument(
        "--output_test_csv",
        type=str,
        default="data/raw/HIV_test.csv"
    )

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.input_csv)
    
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        stratify=data["HIV_active"],
        random_state=42
    )

    # Oversample
    train_data = sampler(train_data)

    # Save
    train_data.to_csv(
        args.output_train_csv,
        index=False
    )
    test_data.to_csv(
        args.output_test_csv,
        index=False
    )