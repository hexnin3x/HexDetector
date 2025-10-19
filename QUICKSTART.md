# HexDetector Quick Start Guide

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hexnin3x/HexDetector.git
   cd HexDetector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure paths**
   
   Edit `src/config/settings.py` and update these paths:
   ```python
   IOT23_SCENARIOS_DIR = "/path/to/your/iot23/scenarios"
   IOT23_ATTACKS_DIR = "/path/to/your/iot23/attacks"
   IOT23_DATA_DIR = "/path/to/your/iot23/data"
   ```

5. **Verify installation**
   ```bash
   python src/utils/check_config.py
   ```

## Quick Demo

Run a quick demo with sample data:

```bash
python src/main.py --mode demo --model random_forest
```

This will:
- Load 10,000 samples per file
- Train a Random Forest model
- Evaluate performance
- Save results to `output/` directory

## Available Commands

### Run different modes
```bash
# Demo mode (fast, 10K samples)
python src/main.py --mode demo

# Full mode (all data, takes hours)
python src/main.py --mode full

# Custom mode with specific sample size
python src/main.py --mode custom --samples 50000
```

### Try different models
```bash
# Random Forest
python src/main.py --model random_forest

# XGBoost
python src/main.py --model xgboost

# SVM
python src/main.py --model svm

# Naive Bayes
python src/main.py --model naive_bayes

# Logistic Regression
python src/main.py --model logistic_regression

# Decision Tree
python src/main.py --model decision_tree

# Gradient Boosting
python src/main.py --model gradient_boosting
```

### Specify output directory
```bash
python src/main.py --mode demo --output ./my_experiment
```

## Dataset Setup

### Download IoT-23 Dataset

1. Visit: https://www.stratosphereips.org/datasets-iot23
2. Download the "lighter version" (8.8 GB compressed, 44 GB uncompressed)
3. Extract to a directory on your system
4. Update paths in `src/config/settings.py`

### Dataset Structure

The IoT-23 dataset should have this structure:
```
iot23/
├── conn/
│   ├── Scenario1/
│   ├── Scenario2/
│   └── ...
```

## Expected Output

After running, you'll find:

```
output/
└── experiment_YYYYMMDD_HHMMSS/
    ├── processed_data.csv
    ├── features.csv
    ├── metrics.csv
    ├── feature_importance.csv
    ├── random_forest_model.pkl
    └── visualizations/
        ├── confusion_matrix.png
        ├── feature_importance.png
        └── roc_curve.png
```

## Common Issues

### Import errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

### Path errors
```bash
# Verify your configuration
python src/utils/check_config.py
```

### Memory issues
```bash
# Use demo mode or reduce samples
python src/main.py --mode custom --samples 10000
```

## Next Steps

1. **Explore the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

2. **Run multiple models**
   ```bash
   # Edit main.py to train multiple models at once
   python src/main.py --mode demo
   ```

3. **Analyze results**
   - Check `output/` directory for results
   - Review confusion matrices and metrics
   - Examine feature importance

## Support

For issues or questions:
- Check IMPLEMENTATION_SUMMARY.md
- Review logs in `logs/hexdetector.log`
- See README.md for detailed documentation

## Performance Tips

- **Start with demo mode** to test everything works
- **Use custom mode** to tune sample size for your hardware
- **Enable GPU** for XGBoost and deep learning models
- **Monitor memory** usage with `htop` or Activity Monitor

---

Happy detecting! 🛡️
