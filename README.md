# Comcast Product Recommender System

A hybrid recommendation system built with LightFM for personalized Comcast product recommendations. This project uses matrix factorization with both user/item features to handle warm-start and cold-start scenarios.

## 📊 Project Overview

- **Course**: INFO 629 - Applied AI (Drexel University)
- **Professor**: Dr. Rosina Weber
- **Author**: Sahil Ailawadi
- **Framework**: LightFM (Hybrid Matrix Factorization with WARP loss)
- **Features**: Customer demographics, usage patterns, product categories, pricing tiers

---

## 🌐 Interactive Web Demo - Streamlit

Try the live web application for interactive recommendations:

Visit: https://ailawadia2.streamlit.app/

**Features:**
- 🆕 **New Customer Mode**: Input demographics and preferences to get personalized recommendations
- 👤 **Existing Customer Mode**: Look up recommendations by customer ID
- 📊 **Grouped Results**: Top pick, add-ons, mobile/bundles
- 💰 **Savings Calculator**: Compare mobile plan costs

---

## 🚀 Code Access via Binder

Launch an interactive Jupyter notebook environment without any local setup:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sahilailawadi/recommender_system_assignment/HEAD?labpath=Ailawadi-A2.ipynb)
https://mybinder.org/v2/gh/sahilailawadi/recommender_system_assignment/HEAD?labpath=Ailawadi-A2.ipynb

**Click the badge above** to:
- Run the notebook in your browser
- Interact with the code cells
- Test recommendations with custom inputs
- No installation required!

**Note**: Binder may take 2-3 minutes to build the environment on first launch.

---

## 📄 View Static HTML Version

For a quick read-only view of the complete analysis:

1. Download the HTML export: [Ailawadi-A2.html](https://github.com/sahilailawadi/recommender_system_assignment/raw/main/Ailawadi-A2.html)
2. Open in any web browser
3. All outputs, visualizations, and results are pre-rendered

**To generate HTML locally:**
```bash
jupyter nbconvert --to html Ailawadi-A2.ipynb
```

---

## 💻 Local Setup

### Prerequisites

**Required**: Python 3.11.x (LightFM is not compatible with Python 3.12+)

Check your Python version:
```bash
python --version
```

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/sahilailawadi/recommender_system_assignment.git
cd recommender_system_assignment
```

2. **Set up Python 3.11 (if needed):**

Using pyenv:
```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

Or using conda:
```bash
conda create -n recsys python=3.11
conda activate recsys
```

3. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Launch Jupyter:**
```bash
jupyter notebook Ailawadi-A2.ipynb
```

---

## 🌐 Interactive Web Demo - Streamlit

Try the live web application for interactive recommendations:

### Option 1: Online Demo (Streamlit Cloud)

Visit: https://ailawadia2.streamlit.app/

**Features:**
- 🆕 **New Customer Mode**: Input demographics and preferences to get personalized recommendations
- 👤 **Existing Customer Mode**: Look up recommendations by customer ID
- 📊 **Grouped Results**: Top pick, add-ons, mobile/bundles
- 💰 **Savings Calculator**: Compare mobile plan costs

---

## 📁 Dataset

The project uses synthetic Comcast customer data:

- **users_v3.csv**: 1000+ customer profiles (age, location, tenure, services)
- **items_v3.csv**: 15 products (internet, TV, phone, bundles, add-ons)
- **interactions_v3.csv**: User-item interaction history

Data generated using `dataGen.py` to simulate realistic customer behavior patterns.

---

## 🧠 Model Details

- **Algorithm**: LightFM (Hybrid Matrix Factorization)
- **Loss Function**: WARP (Weighted Approximate-Rank Pairwise)
- **Components**: 32 latent dimensions
- **Features**:
  - User: Age bins, location, tenure bins, service counts
  - Item: Categories, price bins, tier levels
- **Evaluation Metrics**: Precision@k, Recall@k, F1-Score, AUC

---

## 📝 Project Structure

```
├── Ailawadi-A2.ipynb          # Main analysis notebook
├── streamlit_app.py           # Interactive web application
├── requirements.txt           # Python dependencies
├── dataGen.py                 # Synthetic data generator
├── users_v3.csv              # Customer profiles
├── items_v3.csv              # Product catalog
├── interactions_v3.csv       # Interaction history
└── README.md                 # This file
```

---

## 🔧 Troubleshooting

### Python Version Issues
**Error**: `LightFM installation fails` or `AttributeError: 'dict' object has no attribute 'iteritems'`

**Solution**: Ensure you're using Python 3.11.x:
```bash
python --version  # Should show 3.11.x
```

### LightFM Installation
If `pip install` fails, install from source:
```bash
pip install git+https://github.com/lyst/lightfm.git
```

### OpenMP Warning
**Warning**: `LightFM was compiled without OpenMP support`

This is informational only - the model will use single-threaded execution. Performance impact is minimal for this dataset size.

---

## 📚 References

- [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)
- [WARP Loss Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf)
- [Hybrid Recommender Systems](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_15)

---

## 📧 Contact

**Sahil Ailawadi**
- GitHub: [@sahilailawadi](https://github.com/sahilailawadi)
- Email: sahil.ailawadi@gmail.com

---

## 📄 License

This project is for academic purposes as part of INFO 629 coursework at Drexel University.
