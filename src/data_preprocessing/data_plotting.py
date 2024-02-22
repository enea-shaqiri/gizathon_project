import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(df):
    f, axes = plt.subplots(5, 4, figsize=(24, 20))
    for i, col in enumerate(df.columns):
        if col == "date":
            continue
        sns.histplot(df[col], ax=axes[int(i // 4), int(i % 4)])
    plt.show()