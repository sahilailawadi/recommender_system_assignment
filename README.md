# Comcast Product Recommender System

A hybrid recommendation system built with LightFM for personalized Comcast product recommendations. 
This project uses matrix factorization with both user/item features to handle warm-start and cold-start scenarios.
The purpose of this demo is to illustrate how a telecommunications provider can use a hybrid recommender system to improve product personalization, upsell opportunities, and customer retention.

## 📊 Project Overview

- **Course**: INFO 629 - Applied AI (Drexel University)
- **Professor**: Dr. Rosina Weber
- **Author**: Sahil Ailawadi
- **Framework**: LightFM (Hybrid Matrix Factorization with WARP loss)
- **Features**: Customer demographics, usage patterns, product categories, pricing tiers

---

## How To Run ##

- Binder (no setup) : https://mybinder.org/v2/gh/sahilailawadi/recommender_system_assignment/HEAD?labpath=Ailawadi-A2.ipynb
- Streamlit (Interactive Demo) : https://ailawadia2.streamlit.app/
- Local Jupyter (Python 3.11 required)


## 🌐 Interactive Web Demo - Streamlit

Try the live web application for interactive recommendations.
Streamlit retrains a small model on launch for demo simplicity.

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

- https://mybinder.org/v2/gh/sahilailawadi/recommender_system_assignment/HEAD?labpath=Ailawadi-A2.ipynb

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

**Required**: Python 3.11.x 
(LightFM install from source fails for higher versions of python)

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

## 📁 Dataset

The project uses synthetic Comcast customer data:

- **users_v3.csv**: 1000+ customer profiles (800 Existing Customers and 200 New Customers)
- **items_v3.csv**: 16 products (internet tiers, mobile plans, add-ons, bundles/offers)
- **interactions_v3.csv**: User-item interaction

Data generated using `dataGen_v3.py` to simulate realistic customer behavior patterns.

---

## 🧠 Model Details

- **Algorithm**: LightFM (Hybrid Matrix Factorization)
    This is a hybrid recommender because it combines collaborative filtering (user–item interaction matrix) 
    with extra features for both users and items, enabling cold-start recommendations.
- **Loss Function**: WARP (Weighted Approximate-Rank Pairwise)
- **Components**: 32 latent dimensions
- **Features**:
  - **User Features**:
    - Region
    - Outage risk
    - Household size
    - Devices and IoT count
    - WFH, gamer, and creator counts
    - Budget
    - Broadband usage (GB/month)
    - Mobile line count
    - Mobile data usage
    - Current mobile bill
    - Service flags (has_mobile, is_new_customer)

- **Item Features**:
    - Product category (internet tier, mobile plan, add-on, bundle, offer)
    - Pricing tier
    - Speed level (for internet products)

- **Evaluation Metrics**: 
    - Precision@k and Recall@k (ranking accuracy)
    - F1@k (harmonic balance of precision/recall)
    - AUC (pairwise ranking performance)

---

## 📝 Project Structure

```
├── Ailawadi-A2.ipynb          # Main analysis notebook
├── Ailawadi-A2.html          # HTML for notebook analysis
├── streamlit_app.py           # Interactive web application
├── requirements.txt           # Python dependencies
├── dataGen_v3.py               # Synthetic data generator
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
