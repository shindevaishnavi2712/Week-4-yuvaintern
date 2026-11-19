# Beauty & Wellness Market — Exploratory Data Analysis

A beginner-friendly Python project that explores a sample beauty and wellness sales dataset and creates useful features for customer and business analysis.

## Project objective
Explore sales, product categories, customer purchasing patterns, discounts, ratings and channels. Create engineered variables that could support future customer segmentation and predictive modelling.

## Repository structure
```text
beauty_wellness_eda_project/
├── data/
│   └── beauty_wellness_sales.csv
├── src/
│   └── eda_analysis.py
├── outputs/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset note
The included CSV is **synthetic sample data created for this project**. It is not real customer information and should not be presented as actual market research. Replace it with your approved preprocessed internship dataset if one is provided.

## Features created
- `order_value`: price × quantity − discount
- `discount_rate`: discount divided by gross line value
- `order_month`, `order_day`, `is_weekend`: date-based features
- Customer-level `recency_days`, `frequency`, `monetary`, `average_order_value`, and `category_diversity`

## EDA performed
- Dataset shape, data types, missing values and duplicates
- Order value distribution
- Monthly revenue trend
- Revenue by product category
- Numeric correlation heatmap
- Category-level revenue and order summaries

## Setup and run

1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the analysis:

```bash
python src/eda_analysis.py
```

The script saves cleaned data, customer features, summary tables and charts in `outputs/`.

## Libraries
Python, pandas, NumPy, Matplotlib.

## Future improvements
- Add an interactive dashboard using Streamlit.
- Build RFM customer segments.
- Compare repeat-purchase behaviour.
- Train a model after defining a valid prediction target and time-based validation strategy.

## Author
Add your name here.
