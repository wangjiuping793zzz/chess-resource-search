import csv
import os
import glob
import matplotlib.pyplot as plt


def find_latest_strength_csv():
    files = glob.glob("results/depth_strength_experiment_*.csv")

    if not files:
        raise FileNotFoundError("No depth_strength_experiment CSV file found.")

    return max(files, key=os.path.getmtime)


def read_strength_results(file_path):
    grouped = {}

    with open(file_path, mode="r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            depth = int(row["challenger_depth"])
            score = float(row["challenger_score"])

            if depth not in grouped:
                grouped[depth] = {
                    "games": 0,
                    "score": 0.0
                }

            grouped[depth]["games"] += 1
            grouped[depth]["score"] += score

    depths = []
    score_rates = []

    for depth in sorted(grouped.keys()):
        games = grouped[depth]["games"]
        total_score = grouped[depth]["score"]
        score_rate = total_score / games if games > 0 else 0

        depths.append(depth)
        score_rates.append(score_rate)

    return depths, score_rates


def plot_score_rate(depths, score_rates, output_dir):
    plt.figure()
    plt.plot(depths, score_rates, marker="o")
    plt.xlabel("Challenger Search Depth")
    plt.ylabel("Score Rate Against Baseline")
    plt.title("Search Depth vs Engine Score Rate")
    plt.ylim(0, 1)
    plt.grid(True)

    output_path = os.path.join(output_dir, "depth_vs_score_rate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    csv_file = find_latest_strength_csv()
    print(f"Using CSV file: {csv_file}")

    output_dir = "results/figures_strength"
    os.makedirs(output_dir, exist_ok=True)

    depths, score_rates = read_strength_results(csv_file)
    plot_score_rate(depths, score_rates, output_dir)

    print(f"Figure saved to {output_dir}/depth_vs_score_rate.png")


if __name__ == "__main__":
    main()