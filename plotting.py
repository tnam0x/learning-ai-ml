import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple


# data must be a dictionary
def plot_bar_chart(data: Dict[str, float]):
    categories, values = zip(*data.items())
    plt.figure(figsize=(8, 4))
    plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title('Bar Chart')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()

# data must be 2D array or DataFrame
def plot_heatmap(data: np.ndarray):
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Heatmap')
    plt.tight_layout()
    plt.show()

# x and y must be 1D arrays
def plot_line_chart(x: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='dashed', color='green')
    plt.title('Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# data must be 1D array
def plot_histogram(data: np.ndarray, bins: int = 10):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, color='orange', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# x and y must be 1D arrays
def plot_scatter(x: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, color='red')
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.tight_layout()
    plt.show()


bar_data: Dict[str, float] = {'A': 4, 'B': 7, 'C': 1, 'D': 8}
plot_bar_chart(bar_data)

heatmap_data: np.ndarray = np.random.rand(5, 5)
plot_heatmap(heatmap_data)

x: np.ndarray = np.linspace(0, 10, 10)
y: np.ndarray = np.sin(x)
plot_line_chart(x, y)

hist_data: np.ndarray = np.random.normal(0, 1, 1000)
plot_histogram(hist_data, bins=30)

scatter_x: np.ndarray = np.random.rand(100)
scatter_y: np.ndarray = np.random.rand(100)
plot_scatter(scatter_x, scatter_y)
