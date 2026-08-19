import csv
import os
import glob
import matplotlib.pyplot as plt


def find_latest_multiple_position_csv():
    """
    Find the most recent multiple position experiment CSV file.
    """
    files = glob.glob("results/multiple_position_experiments_*.csv")

    if not files:
        raise FileNotFoundError("No multiple_position_experiments CSV file found in results/.")

    return max(files, key=os.path.getmtime)


def read_results(file_path):
    """
    Read experiment results and group them by position name.
    """
    grouped = {}

    with open(file_path, mode="r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            position = row["position_name"]

            if position not in grouped:
                grouped[position] = []

            grouped[position].append({
                "depth": int(row["depth"]),

                "minimax_nodes": float(row["minimax_nodes"]),
                "alphabeta_nodes": float(row["alphabeta_nodes"]),
                "ordered_nodes": float(row["ordered_nodes"]),

                "minimax_time": float(row["minimax_time"]),
                "alphabeta_time": float(row["alphabeta_time"]),
                "ordered_time": float(row["ordered_time"]),

                "minimax_ebf": float(row["minimax_effective_branching_factor"]),
                "alphabeta_ebf": float(row["alphabeta_effective_branching_factor"]),
                "ordered_ebf": float(row["ordered_effective_branching_factor"]),

                "alphabeta_reduction": float(row["alphabeta_node_reduction_percent"]),
                "ordering_extra_reduction": float(row["ordering_extra_node_reduction_percent"])
            })

    return grouped


def plot_metric_for_each_position(grouped, metric_key, ylabel, title_suffix, output_dir):
    """
    For each position, plot Minimax, Alpha-Beta, and Alpha-Beta + Ordering.
    """

    for position, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["depth"])

        depths = [row["depth"] for row in rows]

        if metric_key == "nodes":
            y1 = [row["minimax_nodes"] for row in rows]
            y2 = [row["alphabeta_nodes"] for row in rows]
            y3 = [row["ordered_nodes"] for row in rows]
        elif metric_key == "time":
            y1 = [row["minimax_time"] for row in rows]
            y2 = [row["alphabeta_time"] for row in rows]
            y3 = [row["ordered_time"] for row in rows]
        elif metric_key == "ebf":
            y1 = [row["minimax_ebf"] for row in rows]
            y2 = [row["alphabeta_ebf"] for row in rows]
            y3 = [row["ordered_ebf"] for row in rows]
        else:
            raise ValueError("Unsupported metric key")

        plt.figure()
        plt.plot(depths, y1, marker="o", label="Minimax")
        plt.plot(depths, y2, marker="o", label="Alpha-Beta")
        plt.plot(depths, y3, marker="o", label="Alpha-Beta + Move Ordering")

        plt.xlabel("Search Depth")
        plt.ylabel(ylabel)
        plt.title(f"{position}: {title_suffix}")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{position}_{metric_key}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_reduction_summary(grouped, output_dir):
    """
    Plot node reduction summaries for each position at the maximum available depth.
    """

    positions = []
    alphabeta_reductions = []
    ordering_extra_reductions = []

    for position, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["depth"])
        last_row = rows[-1]

        positions.append(position)
        alphabeta_reductions.append(last_row["alphabeta_reduction"])
        ordering_extra_reductions.append(last_row["ordering_extra_reduction"])

    plt.figure()
    plt.bar(positions, alphabeta_reductions)
    plt.xlabel("Position")
    plt.ylabel("Alpha-Beta Node Reduction (%)")
    plt.title("Alpha-Beta Node Reduction at Maximum Depth")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")

    output_path = os.path.join(output_dir, "alphabeta_reduction_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.bar(positions, ordering_extra_reductions)
    plt.xlabel("Position")
    plt.ylabel("Extra Node Reduction from Move Ordering (%)")
    plt.title("Move Ordering Extra Node Reduction at Maximum Depth")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")

    output_path = os.path.join(output_dir, "move_ordering_reduction_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    csv_file = find_latest_multiple_position_csv()
    print(f"Using CSV file: {csv_file}")

    output_dir = "results/figures_multiple_positions"
    os.makedirs(output_dir, exist_ok=True)

    grouped = read_results(csv_file)

    plot_metric_for_each_position(
        grouped,
        metric_key="nodes",
        ylabel="Nodes Searched",
        title_suffix="Search Depth vs Nodes Searched",
        output_dir=output_dir
    )

    plot_metric_for_each_position(
        grouped,
        metric_key="time",
        ylabel="Search Time (seconds)",
        title_suffix="Search Depth vs Search Time",
        output_dir=output_dir
    )

    plot_metric_for_each_position(
        grouped,
        metric_key="ebf",
        ylabel="Effective Branching Factor",
        title_suffix="Search Depth vs Effective Branching Factor",
        output_dir=output_dir
    )

    plot_reduction_summary(grouped, output_dir)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()