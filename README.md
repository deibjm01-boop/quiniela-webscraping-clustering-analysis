# Quiniela Analysis: Web Scraping, EDA & Unsupervised Learning

This project analyzes Spanish football betting data (Quiniela) through a data science pipeline:

- Web scraping of real probabilities and results  
- Exploratory Data Analysis (EDA)  
- Unsupervised learning (clustering) to identify match profiles  

The goal is to understand patterns in match outcome probabilities and evaluate the reliability of favorites.

---

## Project Structure

```
quiniela-analysis/
│
├── src/
│   ├── 01_scraping.py        # Data extraction from API
│   ├── 02_eda.py             # Data cleaning, feature engineering and EDA
│   └── 03_clustering.py      # Unsupervised learning (KMeans, GMM, DBSCAN)
│
├── data/                  # Generated datasets (not included)
│   └── .gitkeep
│
├── .gitignore
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Data

The datasets are **not included in the repository**.

They are generated via web scraping from the Eduardo Losilla API.

To reproduce the data:

```bash
python src/scraping.py
```

This will generate:

```
data/quiniela_historico.csv
data/probabilidades_real_2026.csv
```

---

## Methodology

### 1. Web Scraping

Data is collected from:

- Historical Quiniela results (match outcomes, percentages)  
- Real probability distributions (1-X-2)  

Includes:
- retry logic  
- validation checks  
- structured data storage  

---

### 2. Exploratory Data Analysis (EDA)

Main steps:

- Merge historical results with real probabilities  
- Feature engineering:
  - probability differences (`diff_1`, `diff_X`, `diff_2`)  
  - predicted outcome (`signo_probable`)  
  - accuracy of favorite  
- Filtering relevant matchdays (La Liga focus)  
- Analysis of:
  - favorite reliability  
  - draw behavior  
  - probability distributions  

---

### 3. Unsupervised Learning

Clustering is performed using:

- KMeans  
- Gaussian Mixture Models (GMM)  
- DBSCAN  
- Hierarchical clustering (exploratory)  

Supporting techniques:

- PCA for dimensionality reduction and interpretation  
- Elbow method & Silhouette score for cluster selection  

---

## Key Insight

Matches can be grouped into distinct probability profiles, such as:

- clear favorites  
- balanced matches  
- high draw probability matches  

These clusters show different levels of predictability and reliability.

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Set your API token

Create a `.env` file in the root directory:

```
AUTH_TOKEN=your_token_here
```

---

### 3. Run the pipeline

```bash
python src/scraping.py
python src/eda.py
```

Clustering should be executed from a script or notebook after generating `df_liga`.

---

## Technologies Used

- Python  
- pandas, numpy  
- matplotlib  
- scikit-learn  
- scipy  

---

## Author

Project developed by David Jacobs

Data Science applied to football analytics  

---

## Notes

- Data is used for educational and analytical purposes  
- API access requires authentication  
