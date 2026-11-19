from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "beauty_wellness_sales.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    return df

def clean_and_engineer(df):
    df = df.drop_duplicates().copy()
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["discount"] = df["discount"].fillna(0)
    df["order_value"] = df["price"] * df["quantity"] - df["discount"]
    df["discount_rate"] = np.where(
        df["price"] * df["quantity"] > 0,
        df["discount"] / (df["price"] * df["quantity"]), 0
    )
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_day"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def customer_features(df):
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    features = df.groupby("customer_id").agg(
        recency_days=("order_date", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
        category_diversity=("category", "nunique")
    ).reset_index()
    return features

def make_plots(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["order_value"], bins=25, edgecolor="black")
    plt.title("Distribution of Order Value")
    plt.xlabel("Order value (INR)")
    plt.tight_layout()
    plt.savefig(FIG / "order_value_distribution.png", dpi=150)
    plt.close()

    monthly = df.groupby("order_month")["order_value"].sum().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["order_month"], monthly["order_value"], marker="o")
    plt.xticks(rotation=60, ha="right")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue (INR)")
    plt.tight_layout()
    plt.savefig(FIG / "monthly_revenue_trend.png", dpi=150)
    plt.close()

    category = df.groupby("category")["order_value"].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(category["category"], category["order_value"])
    plt.xticks(rotation=25, ha="right")
    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue (INR)")
    plt.tight_layout()
    plt.savefig(FIG / "revenue_by_category.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    corr = df[["price","quantity","discount","rating","order_value","discount_rate"]].corr()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", dpi=150)
    plt.close()

def main():
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\\nMissing values:\\n", df.isna().sum())
    print("\\nDuplicate rows:", df.duplicated().sum())

    clean = clean_and_engineer(df)
    features = customer_features(clean)
    OUT.mkdir(exist_ok=True)
    clean.to_csv(OUT / "cleaned_beauty_wellness_data.csv", index=False)
    features.to_csv(OUT / "customer_features.csv", index=False)
    make_plots(clean)

    summary = clean.groupby("category").agg(
        total_orders=("order_id","nunique"),
        total_revenue=("order_value","sum"),
        average_order_value=("order_value","mean")
    ).sort_values("total_revenue", ascending=False)
    summary.to_csv(OUT / "category_summary.csv")
    print("\\nCategory summary:\\n", summary)
    print("\\nAnalysis complete. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
