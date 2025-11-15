# College Football Game Predictor

A machine-learning system that predicts the outcome of college football matchups using a feature-engineered dataset, advanced classification models, and real-time scraped data. Includes a lightweight frontend website for viewing predictions interactively.

**Live Demo:** *(add link here)*  
**Screenshot:** *(add image + alt tag here)*

---

## How It's Made

### Tech Used
**Python, BeautifulSoup, Pandas, NumPy, Scikit-Learn, XGBoost, Random Forest, HTML, JavaScript**

### Backend / Model Pipeline
The core of this project is a data pipeline that automatically scrapes and cleans college football statistics using **BeautifulSoup**, stores historical and live game data in structured CSVs, and feeds them into ML models.  
Key steps include:

- Web scraping team stats and game results  
- Cleaning, normalizing, and standardizing team names  
- Feature engineering (home/away indicators, offensive/defensive metrics, moving averages, etc.)  
- Training **Random Forest** and **XGBoost** classifiers to output win probabilities  
- Exporting trained models for deployment  
- A continuous update flow to refresh predictions as new data arrives  

### Frontend Website
Built with **HTML and JavaScript**, the website displays upcoming games, model probabilities, confidence levels, and matchup summaries. Prediction outputs are fetched from the backend and rendered in a user-friendly layout suitable for non-technical users.

---

## Optimizations

- Refactored the data pipeline to reduce scraping time and minimize repeated requests  
- Improved model inference speed by pruning features and optimizing preprocessing  
- Added caching for static team information to avoid redundant computation  
- Tuned hyperparameters for Random Forest and XGBoost to improve calibration and accuracy  
- Implemented consistent standardization rules for team names to eliminate mismatches across data sources  

These changes noticeably increased accuracy and stability in real-world usage.

---

## Lessons Learned

Building this predictor involved navigating challenges in:

- Handling inconsistent sports-data formats across different sources  
- Creating a generalizable ML model rather than overfitting to one season  
- Addressing class imbalance, feature leakage, and calibration in sports predictions  
- Deploying ML outputs into a clean, accessible frontend environment  
- Designing a pipeline that updates automatically without breaking due to layout changes on scraped websites  

This project strengthened my skills in data engineering, model interpretability, and robust scraping strategies while working in sports analytics.

