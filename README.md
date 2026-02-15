# Tesla News Sentiment & Stock Direction Prediction

## Project Overview

This project explores whether daily news sentiment and company communications can help predict short-term Tesla (TSLA) stock price movements.

**Core Question:** Can we use natural language processing on financial news and earnings calls to predict whether Tesla's stock will trend upward over the next week?

**Key Objectives:**
- Extract sentiment signals from financial news and earnings communications
- Engineer time-based features that respect temporal dependencies
- Build a predictive model for 5-day forward stock direction
- Maintain rigorous train/test separation to avoid data leakage
- Prioritize interpretability and methodological soundness

---

## Repository Navigation

### Presentation (Main Directory)

**`Tesla_Stock_Prediction_Slideshow.pdf`**
- Short presentation for potential stakeholders.

### Notebooks (Main Directory)

**`obtain_tesla_news.ipynb`**
- Data collection code for news articles, earnings calls, and SEC filings
- Outputs:
  - `data_storage/tesla_daily_news_2024_2025.csv`
  - `earnings_calls/` folder
  - `sec_filings/` folder

**`tesla_sentiment_analysis.ipynb`**
- Sentiment analysis using HuggingFace FinBERT
- Feature engineering (moving averages, lags, interaction terms)
- Statistical analysis (sentiment-volume correlation, A/B testing)

**`tesla_stock_prediction.ipynb`**
- Model training and evaluation (Logistic Regression, XGBoost)
- Train/test splitting with data leakage prevention
- Performance metrics and model comparison

### Data & Outputs

**`data_storage/`** - Raw and processed datasets  
**`earnings_calls/`** - Quarterly earnings call transcripts  
**`sec_filings/`** - SEC 10-K and 10-Q filings  
**`output_plots/`** - Visualization outputs (ROC curves, confusion matrices, feature importance)

---

## Data Collection

### Stock Price & Volume Data
**Source:** Yahoo Finance via `yfinance` Python library
```python
import yfinance as yf
stock_data = yf.download('TSLA', start='2024-01-01', end='2026-01-01')
```

**Coverage:** January 1, 2024 - January 1, 2026  
**Features:** Daily open, high, low, close, volume

---

### News Articles
**Source:** Google News RSS feeds

**Method:** Automated daily scraping using `feedparser` library

**Coverage:** January 2024 - December 2025  
**Volume:**
- Total articles collected: 25,398
- Unique days covered: 732
- Average articles per day: 34.7 (range: 3-92)

---

### Earnings Call Transcripts
**Primary Source:** [The Motley Fool](https://www.fool.com/quote/nasdaq/tsla/)  
**Supplementary:** [Seeking Alpha](https://seekingalpha.com/) (Q1 2025 only)

**Coverage:** 9 quarterly earnings calls (Q4 2023 - Q4 2025)

---

### SEC Filings
**Source:** [SEC EDGAR Database](https://www.sec.gov/edgar)

**Documents Collected:**
- 10-K (Annual Reports): 2023, 2024
- 10-Q (Quarterly Reports): Q1-Q4 2024, Q1-Q4 2025

**Example:** [Tesla 10-Q (Q1 2024)](https://www.sec.gov/Archives/edgar/data/1318605/000162828024017503/tsla-20240331.htm)

**Note:** SEC filings were collected for project completeness but were not used as features in the final model due to time constraints and focus on higher-frequency news signals.

---

## Sentiment Analysis

**Model:** FinBERT (financial domain-adapted BERT)  
**Source:** `ProsusAI/finbert` via HuggingFace Transformers

**Why FinBERT:**
- Pre-trained on financial text (earnings calls, analyst reports, financial news)
- Understands finance-specific language and context
- Returns sentiment classification (positive/negative/neutral) with confidence scores

**Application:**
- **News Headlines:** Analyzed daily, aggregated to daily average sentiment
- **Earnings Transcripts:** Chunked and analyzed, aggregated to quarterly sentiment scores

---

## Feature Engineering

### Daily News Features
- `avg_sentiment`: Daily average sentiment score (-1 to +1)
- `avg_sentiment_ma7`: 7-day moving average of sentiment
- `avg_sentiment_lag1`: Previous day's sentiment (lag feature)
- `article_count`: Number of articles published that day
- `article_count_ma7`: 7-day moving average of article volume
- `article_count_lag1`: Previous day's article count
- `interaction`: Sentiment × volume interaction term
- `interaction_ma7`: 7-day MA sentiment × 7-day MA volume

**Rationale:**
- Moving averages smooth daily noise and capture trends
- Lag features prevent look-ahead bias
- Interaction terms capture amplification effects (high volume + strong sentiment)

### Stock Price Features
- `return`: Daily price return (percent change)
- `return_ma5`: 5-day moving average of returns
- `volume_change`: Daily volume percent change

### Quarterly Features
- `latest_earnings_sentiment`: Most recent earnings call sentiment (forward-filled)

---

## Target Variable

**Target:** Binary classification of 5-day forward price movement
```python
# Calculate 5-day forward return
df['return_5d'] = df['Close'].pct_change(5).shift(-5)

# Target: 1 if price increases, 0 if decreases
df['target'] = (df['return_5d'] > 0).astype(int)
```

**Why 5-day prediction instead of next-day?**
- Daily stock movements are dominated by noise (~50% random)
- 5-day window filters remain actionable
- Sentiment signals take time to propagate through market behavior

---

## Modeling Approach

### Train/Test Split Strategy

**Critical Consideration:** Time-series data with moving averages requires careful splitting to avoid data leakage.

**Implementation:**
- **Training Set:** Jan 1, 2024 - Jan 31, 2025 (257 samples)
- **Buffer Period:** Feb 1-9, 2025 (10 days, discarded)
- **Test Set:** Feb 10, 2025 onwards (225 samples)

**10-day buffer ensures:**
- No overlap between 7-day moving averages in train and test
- Realistic forward-looking evaluation
- No look-ahead bias

**Class Balance:**
- Training: 50.19% up days, 49.81% down days
- Test: 52.44% up days, 47.56% down days

---

### Model Selection

**Primary Model:** Logistic Regression with L2 regularization

**Why Logistic Regression:**
- Strong performance (AUC 0.57) on validation and test sets
- Interpretable coefficients (feature importance clear to stakeholders)
- Well-suited for linear relationships in small datasets
- Computationally efficient

**Alternative Explored:** XGBoost
- Test AUC: 0.52 (underperformed logistic regression)
- Likely reasons: Small dataset (257 training samples), linear feature relationships
- Conclusion: Tree-based models require more data to capture value

---

## Model Performance

**Logistic Regression Results:**

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.57 |
| **Accuracy** | 56% |
| **Precision (Up)** | 57% |
| **Recall (Up)** | 69% |

**Interpretation:**
- AUC 0.57 represents 14% relative improvement over random guessing (0.50)
- Model successfully extracts signal from noisy financial data
- Performance aligns with realistic expectations for stock prediction
- Balanced precision/recall indicates no systematic bias toward up or down predictions

**Why Not Higher?**
- Stock prices are influenced by thousands of factors beyond news sentiment
- Market efficiency theory suggests predictable patterns are quickly arbitraged away
- External shocks (regulatory changes, competitor actions, macroeconomic events) are unpredictable

---

## Key Findings

### Sentiment-Volume Relationship

**Discovery:** High article volume correlates with more negative sentiment

- **Correlation:** -0.25 between 7-day MA article count and sentiment
- **Statistical Significance:** A/B test (p < 0.001) confirms high-volume days have lower sentiment
- **Business Insight:** Major news events (earnings misses, controversies, regulatory actions) generate both high coverage and negative sentiment

### Feature Importance

**Top Predictive Features (from logistic regression coefficients):**
1. Recent price momentum (`return`)
2. Sentiment-volume interaction (`avg_sentiment_ma7`)
3. Article volume changes
4. Closing price

**Insight:** Market momentum dominates, but sentiment provides incremental predictive value.

---

## Limitations & Future Work

### Current Limitations
- Small training set (257 samples) limits model complexity
- Single company focus (Tesla) - findings may not generalize
- News-only sentiment - excludes social media, analyst reports, options flow
- No causality claims - correlation ≠ causation
- Market efficiency - any exploitable pattern degrades over time

### Potential Enhancements
- Larger datasets: Multi-year history, multiple companies
- Additional features: Social media sentiment, insider trading, short interest
- Advanced NLP: Topic modeling, event detection
- Ensemble methods: Combine multiple models for robust predictions

---

## Technical Stack

**Languages & Core Libraries:**
- Python 3.10
- pandas, numpy (data manipulation)
- scikit-learn (modeling, evaluation)
- transformers, torch (NLP)

**Data Collection:**
- yfinance (stock data)
- feedparser (RSS parsing)
- Manual collection (earnings transcripts, SEC filings)

**Visualization:**
- matplotlib, seaborn

---

## Reproducibility

All code is documented with inline comments explaining methodology and rationale. Data collection scripts include timestamps and source attribution.

**To replicate:**
1. Run data collection scripts (note: news data is time-dependent)
2. Execute sentiment analysis notebooks
3. Run feature engineering pipeline
4. Train and evaluate models
