import numpy as np
import matplotlib.pyplot as plt

# 1. Define the data points
x_data = np.array([-7.40, -4.90, -2.40, 0.10, 2.50, 5.00, 7.40, 9.80])
y_data = np.array([-5.30, -2.90, -0.70, 1.40, 3.20, 5.50, 7.30, 8.90])

# 2. Calculate Pearson's correlation coefficient
# np.corrcoef returns a matrix; we need the element at [0, 1] or [1, 0]
correlation_matrix = np.corrcoef(x_data, y_data)
pearson_r = correlation_matrix[0, 1]

print(f"Calculated Pearson's Correlation Coefficient: {pearson_r:.4f}")

# 3. Generate the Graph
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', s=100, label='Data Points', alpha=0.7)

# Optional: Plot a trendline to visualize the correlation
m, b = np.polyfit(x_data, y_data, 1)
plt.plot(x_data, m*x_data + b, color='red', linestyle='--', label=f'Trendline (y={m:.2f}x+{b:.2f})')

# Graph labels and styling
plt.title(f"Scatter Plot with Pearson Correlation (r = {pearson_r:.4f})", fontsize=14)
plt.xlabel("X Coordinates", fontsize=12)
plt.ylabel("Y Coordinates", fontsize=12)
plt.axhline(0, color='blue', linewidth=0.5)
plt.axvline(0, color='blue', linewidth=0.5)
plt.grid(color='gray', linestyle='-.', linewidth=0.5)
plt.legend()

# Show the graph
plt.show()