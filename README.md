# College Football Game Predictor

A machine-learning system that predicts the outcome of college football matchups using a feature-engineered dataset, advanced classification models, and real-time scraped data. Includes a lightweight frontend website for viewing predictions interactively.

**Live Demo:** *((https://predictorfile.onrender.com))*  

---

## How It's Made

### Tech Used
**Python, BeautifulSoup, Pandas, NumPy, Scikit-Learn, Tensorflow, XGBoost, Random Forest, HTML, JavaScript**

### Backend / Model Pipeline
The core of this project is a data pipeline that automatically scrapes and cleans college football statistics using **BeautifulSoup**, stores historical and live game data in structured CSVs, and feeds them into ML models.  
Key steps include:

- Web scraping team stats and game results  
- Cleaning, normalizing, and standardizing team names  
- Feature engineering (home/away indicators, offensive/defensive metrics, etc.)  
- Training **Random Forest** and **XGBoost** classifiers to output win probabilities  
- Models then used in a stacked model, extracting the meta features to train a feed-forward neural network


### Frontend Website
Built with **HTML and JavaScript**, the website allows users to input teams and see the predicted winner along with the win probability. Prediction outputs are fetched from the backend and rendered in a user-friendly layout suitable for non-technical users.

---

## Optimizations

- Refactored the data pipeline to reduce scraping time and minimize repeated requests  
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

