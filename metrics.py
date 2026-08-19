def calculate_approximate_effective_branching_factor(
    nodes: int,
    depth: int,
) -> float:
    """
    Calculate the approximate effective branching factor.

    This project uses the simplified approximation:

        b_hat = nodes ** (1 / depth)

    The result is not the standard effective branching factor obtained by
    solving:

        nodes = 1 + b + b^2 + ... + b^depth

    Args:
        nodes:
            Number of searched nodes. The current project excludes the root
            node from this count.
        depth:
            Nominal search depth in plies.

    Returns:
        The approximate effective branching factor.

    Raises:
        ValueError:
            If nodes is negative or depth is not positive.
    """

    if nodes < 0:
        raise ValueError("nodes must not be negative.")

    if depth <= 0:
        raise ValueError("depth must be greater than zero.")

    if nodes == 0:
        return 0.0

    return nodes ** (1.0 / depth)


def calculate_node_reduction(
    baseline_nodes: int,
    method_nodes: int,
) -> float:
    """
    Calculate the percentage reduction in searched nodes.

    The calculation is:

        reduction = 100 * (baseline_nodes - method_nodes) / baseline_nodes

    Positive values indicate that the evaluated method searched fewer nodes
    than the baseline. A negative value indicates that it searched more nodes.

    Args:
        baseline_nodes:
            Number of nodes searched by the reference method.
        method_nodes:
            Number of nodes searched by the evaluated method.

    Returns:
        Node-reduction percentage.

    Raises:
        ValueError:
            If either node count is negative or if baseline_nodes is zero.
    """

    if baseline_nodes < 0 or method_nodes < 0:
        raise ValueError("Node counts must not be negative.")

    if baseline_nodes == 0:
        raise ValueError("baseline_nodes must be greater than zero.")

    return (
        100.0
        * (baseline_nodes - method_nodes)
        / baseline_nodes
    )


def calculate_effective_branching_factor(
    nodes: int,
    depth: int,
) -> float:
    """
    Backward-compatible alias for older experiment scripts.

    New code should call
    calculate_approximate_effective_branching_factor() explicitly.
    """

    return calculate_approximate_effective_branching_factor(nodes, depth)
