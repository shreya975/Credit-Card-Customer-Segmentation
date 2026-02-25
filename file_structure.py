import os

base_dir = "fraud-detection-system"

# Folders to create
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/model",
    "src/monitoring",
    "src/utils",
    "artifacts/models",
    "artifacts/scalers",
    "artifacts/reports",
    "logs",
    "app",
]

# Files to create (with optional content)
files = {
    "notebooks/01_eda.ipynb": "",
    "notebooks/02_feature_validation.ipynb": "",
    
    "src/__init__.py": "",
    "src/config.py": "",
    "src/data_loader.py": "",
    "src/feature_engineering.py": "",
    "src/leakage_safe_aggregations.py": "",
    
    "src/model/__init__.py": "",
    "src/model/train.py": "",
    "src/model/evaluate.py": "",
    "src/model/threshold_optimization.py": "",
    "src/model/explainability.py": "",
    
    "src/monitoring/__init__.py": "",
    "src/monitoring/drift.py": "",
    "src/monitoring/metrics_tracker.py": "",
    
    "app/main.py": "# FastAPI main entry\n",
    "app/schema.py": "# Request schema definitions\n",
    
    "Dockerfile": "",
    "requirements.txt": "",
    "README.md": "# Fraud Detection System\n",
    ".gitignore": "",
}

# Create folders
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# Create files
for file_path, content in files.items():
    full_path = os.path.join(base_dir, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print("✔ File structure created successfully!")