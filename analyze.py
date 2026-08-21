import pandas as pd
import matplotlib.pyplot as plt
from data import get_filtered_data

def main():

    series = get_filtered_data()
    if not series:
        print("No data available for analysis.")
        return
    totals = pd.Series({keyword: ser.sum() for keyword, ser in series.items()}).nlargest(10)

    # Visualize the top 10 keywords over time with their distinct video counts as a line chart
    plt.figure(figsize=(12, 6))
    for keyword in totals.index:
        plt.plot(series[keyword].index, series[keyword].values, label=keyword)
    plt.title("Top 10 Trending Keywords Over Time")
    plt.xlabel("Time")
    plt.ylabel("Distinct Videos")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("top10_trends_over_time.png")
    plt.close()

    # Visualize the top 10 keywords with their total distinct video counts as a bar chart
    plt.figure(figsize=(10, 6))
    totals.plot(kind='barh', title="Top 10 Trending Keywords by Total Distinct Videos", legend=False)
    plt.xlabel("Total Distinct Videos")
    plt.ylabel("Keywords")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig("top10_trends_bar_chart.png")
    plt.close()

    print("Analysis completed. Visualizations saved as 'top10_trends_over_time.png' and 'top10_trends_bar_chart.png'.")

if __name__ == "__main__":
    main()
