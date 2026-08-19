# Resource-Constrained Chess Game-Tree Search

This repository contains the source code, experimental framework, validation tests, and selected results for an MSc project on resource-constrained game-tree search in chess.

The project investigates how computational resource constraints affect both search complexity and playing performance. Chess is used as a controlled experimental platform to compare classical game-tree search techniques under fixed-depth and time-limited conditions. The purpose is not to build a state-of-the-art chess engine, but to study the relationships among search depth, time budget, pruning, move ordering, node expansion, effective branching factor, and engine performance.

## Research Question

**How do computational resource constraints influence both the complexity of game-tree search and the performance of a chess engine?**

The investigation focuses on:

- search depth and time budget;
- nodes searched and elapsed search time;
- effective branching factor;
- Alpha-Beta cutoffs and node reduction;
- the additional effect of move ordering;
- playing performance against a fixed baseline;
- fixed and search-aware time-allocation strategies.

## Implemented Methods

The repository includes independently implemented versions of:

- Minimax;
- Alpha-Beta pruning;
- Alpha-Beta pruning with move ordering;
- time-limited search;
- resource-aware time allocation;
- search-aware time allocation.

The move-ordering strategy prioritises tactically relevant moves such as captures, promotions, and checks. The evaluation function considers material balance, piece activity, development, centre control, and mobility. Positive scores indicate an advantage for White, while negative scores indicate an advantage for Black.

## Technologies

- Python 3.13;
- [`python-chess`](https://python-chess.readthedocs.io/) for board representation, legal move generation, and game-state handling;
- Matplotlib for figure generation;
- CSV and text files for experiment outputs and logs.

## Repository Structure

```text
chess-resource-search/
|
|-- README.md
|-- RUN_INSTRUCTIONS_CN.md
|-- TIME_ALLOCATION_EXPERIMENT.md
|-- requirements.txt
|-- .gitignore
|
|-- main.py
|-- evaluation.py
|-- search.py
|-- metrics.py
|-- time_limited_search.py
|-- search_aware_time_limited.py
|
|-- experiments.py
|-- match_runner.py
|-- match_runner_time_allocation.py
|-- match_runner_search_aware.py
|
|-- plot_results.py
|-- plot_multiple_positions.py
|-- plot_strength_results.py
|
|-- validation_tests.py
|-- test_time_allocation.py
|-- test_search_aware_time_allocation.py
|
|-- experiment_log.txt
`-- results/
```

## File Descriptions

### Core search and evaluation

- **`main.py`** runs a quick comparison of Minimax, Alpha-Beta pruning, and Alpha-Beta pruning with move ordering on a chess position. It reports the selected move, evaluation score, nodes searched, elapsed time, effective branching factor, and relevant cutoff statistics.
- **`evaluation.py`** contains the heuristic board-evaluation function used by the engines.
- **`search.py`** contains the fixed-depth Minimax and Alpha-Beta implementations, together with move ordering.
- **`metrics.py`** provides experiment metrics such as effective branching factor and node-reduction percentages.
- **`time_limited_search.py`** provides search functionality controlled by a time budget rather than only by a fixed depth.
- **`search_aware_time_limited.py`** implements the search-aware time-limited strategy used in the later resource-allocation experiments.

### Experiment runners

- **`experiments.py`** compares the fixed-depth search algorithms across multiple representative chess positions and records complexity and efficiency metrics.
- **`match_runner.py`** runs playing-strength experiments between engines using different fixed search depths.
- **`match_runner_time_allocation.py`** evaluates engine behaviour under the time-allocation configuration.
- **`match_runner_search_aware.py`** evaluates the search-aware allocation strategy against the relevant baseline configuration.

### Plotting

- **`plot_results.py`** generates figures for single-position depth experiments.
- **`plot_multiple_positions.py`** generates figures for the multiple-position complexity experiments.
- **`plot_strength_results.py`** generates score-rate figures for playing-strength experiments.

### Validation and documentation

- **`validation_tests.py`** validates the core evaluation, search, and metric behaviour.
- **`test_time_allocation.py`** tests the time-allocation implementation.
- **`test_search_aware_time_allocation.py`** tests the search-aware allocation behaviour.
- **`experiment_log.txt`** records experiment configurations, output files, and observations.
- **`TIME_ALLOCATION_EXPERIMENT.md`** provides additional details about the time-allocation experiments.
- **`RUN_INSTRUCTIONS_CN.md`** provides supplementary execution instructions in Chinese.
- **`results/`** contains selected experiment outputs and generated figures used in the project analysis.

## Experimental Design

### 1. Fixed-Depth Search-Complexity Experiments

Minimax, Alpha-Beta pruning, and Alpha-Beta pruning with move ordering are compared at increasing search depths. The main measurements are:

- nodes searched;
- elapsed search time;
- effective branching factor;
- number of Alpha-Beta cutoffs;
- node reduction from Alpha-Beta pruning;
- additional node reduction from move ordering.

The position suite includes:

- the standard starting position;
- an Open Game position;
- a Ruy Lopez position;
- a Queen's Pawn position.

### 2. Fixed-Depth Playing-Strength Experiments

Engines using different search configurations or depths play matches from selected opening positions. Each configuration is tested with both colours where applicable. The primary performance measure is score rate against a fixed baseline, based on wins, draws, and losses.

### 3. Time-Limited and Resource-Allocation Experiments

The later experiments replace a purely fixed-depth comparison with explicit time constraints. They examine how a limited resource budget is allocated and whether a search-aware allocation strategy changes engine behaviour or playing performance relative to the baseline allocation approach.

## Installation

The project was developed and tested with Python 3.13.

1. Clone or download the repository.
2. Open a terminal in the repository root.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment.

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

5. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The local `.venv/` directory is intentionally excluded from the repository. Each user should create a new environment using the commands above.

## Recommended Reproduction Order

All commands below should be run from the repository root.

### 1. Validate the implementation

```bash
python validation_tests.py
python test_time_allocation.py
python test_search_aware_time_allocation.py
```

### 2. Run a quick fixed-depth comparison

```bash
python main.py
```

This provides a short comparison of the principal fixed-depth search algorithms and confirms that the main search pipeline is operational.

### 3. Run the multiple-position complexity experiment

```bash
python experiments.py
```

The script evaluates the fixed-depth search methods across the position suite and saves the resulting measurements under `results/`.

### 4. Generate complexity figures

```bash
python plot_results.py
python plot_multiple_positions.py
```

### 5. Run the fixed-depth playing-strength experiment

```bash
python match_runner.py
```

Deeper match configurations can require substantially more time than the quick comparison and validation scripts.

### 6. Run the time-allocation experiments

```bash
python match_runner_time_allocation.py
python match_runner_search_aware.py
```

Additional implementation checks can be run separately with:

```bash
python test_time_allocation.py
python test_search_aware_time_allocation.py
```

### 7. Generate playing-strength figures

```bash
python plot_strength_results.py
```

Experiment settings such as search depths, test positions, time budgets, and match configurations are defined in the relevant runner scripts. Selected final outputs are retained in `results/`, while `experiment_log.txt` records the corresponding experiment context.

## Output and Metrics

Depending on the selected experiment, outputs include:

- the best move and heuristic evaluation score;
- nodes searched and search time;
- effective branching factor;
- Alpha-Beta cutoff counts;
- pruning and move-ordering reduction percentages;
- match result and score rate;
- time-budget and allocation measurements;
- CSV result files and generated figures.

Timing measurements are hardware- and system-dependent. Small differences in elapsed time, and therefore in some time-limited outcomes, may occur across machines. Node counts from equivalent fixed-depth configurations provide a more hardware-independent comparison of search complexity.

## Scope and Limitations

This software is a research prototype designed for controlled experimental comparison. It is not intended to compete with production chess engines such as Stockfish. Playing strength is constrained by the deliberately interpretable evaluation function, the tested search depths and time budgets, and the selected position and match suites. Conclusions should therefore be interpreted within the experimental settings documented in the repository and dissertation.

## Implementation Note

`python-chess` is used for chess rules, board representation, legal move generation, and game-state handling. The search algorithms, heuristic evaluation, complexity metrics, time-allocation mechanisms, match runners, and experimental framework used for the study are implemented within this project.

## Academic Context

This repository accompanies an MSc project in Advanced Computer Science at the University of Leeds. It is intended to provide the source code and reproducible experimental material associated with the dissertation.
