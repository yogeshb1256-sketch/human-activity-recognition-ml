import matplotlib.pyplot as plt

# Model names and their corresponding accuracies
models = ['I3D MODEL', 'C3D', 'TSM', 'SlowFast', 'ResNet+LSTM']
accuracies = [69.71, 55.7, 58.2, 60.5, 57.9]

# Custom colors for better style
colors = ['#3B2C85', '#3B6E8F', '#338B8B', '#4BAA87', '#8AC96D']

# Create bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=colors)

# Rotate x labels for style
plt.xticks(rotation=25)
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.title('Comparison of Different HAR Models')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.1f}%', ha='center', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()
