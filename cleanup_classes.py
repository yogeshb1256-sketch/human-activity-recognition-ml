import os

# ✅ List of 5 selected activity classes
selected_classes = ['Archery', 'Basketball', 'BenchPress', 'Bowling', 'PushUps']

# ✅ Full paths to the processed train and validation folders
folders = ['D:/NEW/processed/train', 'D:/NEW/processed/val']

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            # Extract the class label from the filename
            label = filename.split('_')[0]
            if label not in selected_classes:
                filepath = os.path.join(folder, filename)
                os.remove(filepath)
                print(f"Deleted: {filepath}")
            else:
                print(f"Kept: {filename}")

print("\nCleanup complete! Only selected classes remain.")
