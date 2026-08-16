# 🚀 Demo Projects & PoCs

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> A centralized repository for building, testing, and tracking **Proof-of-Concepts (PoCs)**, rapid prototypes, and initial technical demos before deploying them into full Minimum Viable Products (MVPs).

---

## 📌 Repository Overview

This repository serves as a testing laboratory for engineering workflows, machine learning models, and software architectures. Each subfolder represents a standalone demo project equipped with its own modular source code, sample datasets, and documentation.

---

## 🗂️ Demo Projects Directory

| Project Name | Domain / Tech Stack | Target MVP | Status | Link |
| :--- | :--- | :--- | :---: | :---: |
| **ML-Decline-Curve-Analysis** | Python • Streamlit • Scikit-Learn • Plotly | Streamlit DCA Web App | 🟡 In Progress | [View Project](./ML-Decline-Curve-Analysis) |
| **Demo-Project-02** | *Tech Stack* | *Target Application* | ⚪ Planned | [View Project](./Demo-Project-02) |
| **Demo-Project-03** | *Tech Stack* | *Target Application* | ⚪ Planned | [View Project](./Demo-Project-03) |

---

## 📂 Project Structure

```text
Demo-Projects/
│
├── README.md                          # Repository overview and master navigation
├── .gitignore                         # Master git ignore rules
│
├── ML-Decline-Curve-Analysis/         # Demo 01: Machine Learning DCA Tool
│   ├── README.md                      # Detailed project documentation
│   ├── app.py                         # Streamlit dashboard
│   ├── requirements.txt               # Dependencies
│   ├── data/                          # Sample production CSVs
│   └── src/                           # Preprocessing & MLP ML engine
│
└── [Future-Demo-Folder]/              # Demo 02...