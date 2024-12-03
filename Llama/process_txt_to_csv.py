import os
import pandas as pd

# Path to your main data directory
data_dir = "./data/categories"

# Initialize a list to store processed data
data = []

# Walk through all categories and files
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(category_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract the title and content (custom parsing as per your format)
                title = content.split("\n")[0].replace("Title: ", "").strip()
                text = content.split("Content:")[1].strip() if "Content:" in content else content.strip()
                
                # Append to the dataset
                data.append({
                    "category": category,
                    "title": title,
                    "text": text
                })

# Convert to a DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

# Save to CSV files
os.makedirs("./processed_data", exist_ok=True)
train_df.to_csv("./processed_data/train.csv", index=False)
val_df.to_csv("./processed_data/val.csv", index=False)

print("Processed data saved to './processed_data/train.csv' and './processed_data/val.csv'")
