# Machine Learning-Based Decline Curve Analysis (DCA-ML)

An end-to-end Python pipeline and web application designed to forecast oil and gas production using machine learning and deep learning models. This repository implements data preprocessing, feature engineering, neural network sequence modeling, and interactive visualization for field production data.

---

## 📌 Project Overview

Traditional Decline Curve Analysis (DCA)—such as Arps' empirical equations—often struggles with complex operational dynamics, frequent well shut-ins, and multi-variable dependencies (e.g., choke variations and tubing head pressure). 

This project leverages data-driven models (including LSTMs and multi-layer perceptrons) to learn reservoir rate decline trends directly from active production history while incorporating operational parameters like Tubing Head Pressure (THP) and choke sizes.

### Key Features
- **Zero-Production Filtering:** Preprocessing logic to drop non-producing/shut-in periods, allowing models to learn pure reservoir decline physics.
- **Physics-Informed Feature Engineering:** Automatic computation of cumulative production ($N_p$) and normalized producing time ($t/t_{max}$).
- **Sequence Transformation:** Sliding-window tensor generation designed for recurrent time-series models.
- **Multi-Axis Production Plotting:** Custom visualization utilities built with Matplotlib for synchronized $2 \times 1$ production performance tracking (Oil Rate, Cum Oil, GOR, Water Cut, THP, and Choke Size).

---

## 📁 Repository Structure

```text
├── data/
│   └── raw_production.csv       # Field production dataset
├── preprocess.py                # Preprocessing pipeline & sequence generator
├── plot_utils.py                # Custom production plotting routines
├── train.py                     # Model training script
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation