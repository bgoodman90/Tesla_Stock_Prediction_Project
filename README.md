# Tesla News Sentiment & Stock Direction Prediction

## Project Overview

This project explores whether daily news sentiment can help predict 
short-term stock price direction for Tesla (TSLA).

The objective was to build a minimum viable predictive system that:

- Extracts sentiment from financial news articles
- Engineers time-based features
- Predicts 5-day forward stock direction
- Evaluates performance using proper time-series methodology
- Avoids data leakage

This project emphasizes modeling discipline, chronological validation, and 
interpretability over model complexity.

---

## Data Sources

### 1. Daily News Articles
- Sentiment extracted using a transformer-based sentiment model
- Aggregated daily metrics:
  - Average sentiment
  - Article count
  - Positive/negative ratios

### 2. Earnings Call Transcripts
- Earnings calls were parsed and sentiment scored
- Used as a quarterly-level feature (prototype only)
- For this MVP, focus was placed primarily on daily news sentiment

### 3. Stock Price Data
- Source: `yfinance`
- Daily OHLCV data for TSLA
- Target constructed from forward 5-day returns

---

## Target Variable

Instead of predicting next-day movement (too noisy), the final model 
predicts:

> **Whether the stock price will be higher 5 trading days in the future**


