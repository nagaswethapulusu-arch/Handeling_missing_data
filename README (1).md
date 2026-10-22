# 🧹 Handle Missing Values (Pandas)

Handle missing values in a dataset using appropriate Pandas techniques and explain why each method was chosen. A first step in data preprocessing for machine learning.

## Objective
Learn basic data preprocessing for machine learning.

## Tools
- Python
- Pandas
- Jupyter Notebook
- Streamlit (for the deployed demo)

## Dataset
A Titanic-style passenger dataset (`passengers.csv`, 200 rows) with realistic missing values, generated inside the notebook:
`passenger_id, survived, pclass, sex, age, fare, embarked, cabin`

| Column | Missing | Treatment | Why |
|---|---|---|---|
| `cabin` | about 77% | Drop the column | Too many values missing to fill reliably |
| `age` | about 21% | Fill with **median** | Age is right-skewed; median ignores extreme values |
| `fare` | 2% | Fill with **median per class** | Fare depends strongly on passenger class |
| `embarked` | 1% | Fill with **mode** | Categorical column, only 2 rows missing |

A blind `dropna()` would remove about 80% of the rows, so each column is treated separately.

## Project Structure
```
├── Handle_Missing_Values.ipynb   # Main notebook
├── passengers.csv                # Dataset (optional, the notebook creates it)
├── app.py                        # Streamlit web app
├── requirements.txt              # Dependencies
└── README.md
```

## Cheat Sheet

| Goal | Code |
|---|---|
| Count missing per column | `df.isnull().sum()` |
| Percent missing | `df.isnull().mean() * 100` |
| Drop rows with any missing | `df.dropna()` |
| Drop a column | `df.drop(columns=['cabin'])` |
| Fill with median | `df['age'] = df['age'].fillna(df['age'].median())` |
| Fill with mode | `df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])` |
| Fill by group | `df['fare'].fillna(df.groupby('pclass')['fare'].transform('median'))` |

## Run Locally
```bash
git clone https://github.com/<your-username>/pandas-handle-missing-values.git
cd pandas-handle-missing-values
pip install -r requirements.txt

jupyter notebook Handle_Missing_Values.ipynb   # notebook
streamlit run app.py                           # web app
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to share.streamlit.io and click **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

The app includes its own sample data, and you can also upload any CSV, choose a method per column and download the cleaned file.

## Interview Questions
- **Two common ways to handle missing values?** Remove them (`dropna()`) or fill them (`fillna()` with mean, median, mode or a constant).
- **When should you use median instead of mean?** When the column is skewed or has outliers, because the median is not affected by extreme values.
- **When is dropping rows appropriate?** When only a very small share of rows is affected, or when the missing column is essential (such as the target). Avoid it when it would remove a large part of the data.

## Author
Swetha, AI & ML Track, Veda Technology
