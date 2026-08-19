import csv
import os
import matplotlib.pyplot as plt


def read_results(file_path):
    depths = []
    minimax_nodes = []
    alphabeta_nodes = []
    minimax_time = []
    alphabeta_time = []
    minimax_ebf = []
    alphabeta_ebf = []
    node_reduction = []

    with open(file_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            depths.append(int(row["depth"]))

            minimax_nodes.append(float(row["minimax_nodes"]))
            alphabeta_nodes.append(float(row["alphabeta_nodes"]))

            minimax_time.append(float(row["minimax_time"]))
            alphabeta_time.append(float(row["alphabeta_time"]))

            minimax_ebf.append(float(row["minimax_effective_branching_factor"]))
            alphabeta_ebf.append(float(row["alphabeta_effective_branching_factor"]))

            node_reduction.append(float(row["node_reduction_percent"]))

    return {
        "depths": depths,
        "minimax_nodes": minimax_nodes,
        "alphabeta_nodes": alphabeta_nodes,
        "minimax_time": minimax_time,
        "alphabeta_time": alphabeta_time,
        "minimax_ebf": minimax_ebf,
        "alphabeta_ebf": alphabeta_ebf,
        "node_reduction": node_reduction
    }


def save_line_chart(x, y1, y2, xlabel, ylabel, title, label1, label2, output_path):
    plt.figure()
    plt.plot(x, y1, marker="o", label=label1)
    plt.plot(x, y2, marker="o", label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_single_line_chart(x, y, xlabel, ylabel, title, label, output_path):
    plt.figure()
    plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    file_path = "results/depth_experiments.csv"
    output_dir = "results/figures"

    os.makedirs(output_dir, exist_ok=True)

    data = read_results(file_path)

    save_line_chart(
        data["depths"],
        data["minimax_nodes"],
        data["alphabeta_nodes"],
        "Search Depth",
        "Nodes Searched",
        "Search Depth vs Nodes Searched",
        "Minimax",
        "Alpha-Beta",
        f"{output_dir}/depth_vs_nodes.png"
    )

    save_line_chart(
        data["depths"],
        data["minimax_time"],
        data["alphabeta_time"],
        "Search Depth",
        "Search Time (seconds)",
        "Search Depth vs Search Time",
        "Minimax",
        "Alpha-Beta",
        f"{output_dir}/depth_vs_time.png"
    )

    save_line_chart(
        data["depths"],
        data["minimax_ebf"],
        data["alphabeta_ebf"],
        "Search Depth",
        "Effective Branching Factor",
        "Search Depth vs Effective Branching Factor",
        "Minimax",
        "Alpha-Beta",
        f"{output_dir}/depth_vs_effective_branching_factor.png"
    )

    save_single_line_chart(
        data["depths"],
        data["node_reduction"],
        "Search Depth",
        "Node Reduction (%)",
        "Node Reduction from Alpha-Beta Pruning",
        "Node Reduction",
        f"{output_dir}/node_reduction.png"
    )

    print("Figures saved successfully.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()