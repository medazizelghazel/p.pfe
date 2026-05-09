def print_comparison_table():
    cnn1d = {
        "accuracy": 0.5625,
        "macro_precision": 0.4542,
        "macro_recall": 0.4466,
        "macro_f1": 0.3938,
    }

    cnn2d = {
        "accuracy": 0.6167,
        "macro_precision": 0.5265,
        "macro_recall": 0.5456,
        "macro_f1": 0.5301,
    }

    print("\n=== MODEL COMPARISON ===")
    print(f"{'Metric':<20} {'CNN1D':<12} {'CNN2D':<12}")
    print("-" * 46)
    print(f"{'Accuracy':<20} {cnn1d['accuracy']:<12.4f} {cnn2d['accuracy']:<12.4f}")
    print(f"{'Macro Precision':<20} {cnn1d['macro_precision']:<12.4f} {cnn2d['macro_precision']:<12.4f}")
    print(f"{'Macro Recall':<20} {cnn1d['macro_recall']:<12.4f} {cnn2d['macro_recall']:<12.4f}")
    print(f"{'Macro F1':<20} {cnn1d['macro_f1']:<12.4f} {cnn2d['macro_f1']:<12.4f}")

    print("\nConclusion:")
    if cnn2d["macro_f1"] > cnn1d["macro_f1"]:
        print("CNN2D est meilleur que CNN1D selon le Macro F1-score.")
    elif cnn2d["macro_f1"] < cnn1d["macro_f1"]:
        print("CNN1D est meilleur que CNN2D selon le Macro F1-score.")
    else:
        print("CNN1D et CNN2D ont le même Macro F1-score.")


if __name__ == "__main__":
    print_comparison_table()