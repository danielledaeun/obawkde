# Online Biased Learning with Adaptive Weight Kernel Density Estimation

This project implements an online learning approach for handling class imbalance in streaming data using Adaptive Weight Kernel Density Estimation (AWKDE). The implementation includes various baseline methods and comparative analysis tools.

## Prerequisites

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/danielledaeun/obawkde.git
cd obawkde
```

2. Set up Python environment:
```bash
conda create --name obawkde python=3.12
conda activate obawkde
pip install -r requirements.txt
```

3. Install the AWKDE package:
```bash
git submodule update --init
cd lib/awkde
conda install -c conda-forge compilers  # Required for compilation on some systems
pip install -e .
cd ../..
```

## Project Structure
```
obawkde/
├── data/             # Directory for generated datasets
├── lib/
│   └── awkde/        # AWKDE implementation
├── notebooks/        # Analysis notebooks
│   ├── generate_data.ipynb    # Data generation scripts
│   ├── table_results.ipynb    # Results analysis
│   └── figure_results.ipynb   # Visualization tools
├── main.py           # Main execution script
├── proposing.py      # Proposed method implementation
├── base.py         # Base models for online learning
└── README.md
```

## Data Generation

Before running experiments, you need to generate the required datasets:

1. Create necessary data files:
```bash
jupyter notebook notebooks/generate_data.ipynb
```

This will generate various synthetic datasets (sea, sine, circle) with different characteristics in the `data/` directory.

Note: While the `data/` directory is maintained in the repository structure, the generated `.csv` files are not tracked by git.

## Usage

1. Generate data (if not already done):
```bash
jupyter notebook notebooks/generate_data.ipynb
```

2. Execute experiments:
```bash
python main.py
```

3. Analyze results:
```bash
jupyter notebook notebooks/table_results.ipynb
```

## Results

Results are stored in the `res/` directory with the following structure:
```
res/
├── sea/
│   ├── noise/
│   │   ├── 0/     # Imbalance Ratio: 0.1%
│   │   ├── 1/     # Imbalance Ratio: 1%
│   │   └── 10/    # Imbalance Ratio: 10%
│   ├── safe/
│   └── borderline/
```

Note: Result files are not included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.