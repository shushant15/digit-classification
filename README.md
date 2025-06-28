# Digit Classification
Uses Scikit-learn’s digits dataset to classify handwritten digits (0–9).
Main script: plot_digits_classification.py – calls functions only.
Logic (load, visualize, preprocess, train, predict, evaluate) moved to utils.py.

conda create -n digits python=3.13
conda activate digits
pip install -r requirements.txt
python plot_digits_classification.py
