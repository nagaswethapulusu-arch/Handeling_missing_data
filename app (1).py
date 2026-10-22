import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Handle Missing Values", page_icon="🧹", layout="wide")
st.title("🧹 Handle Missing Values with Pandas")
st.caption("Upload a CSV (or use the built-in Titanic-style sample), analyse the missing values, choose a method for each column and download the cleaned data.")


@st.cache_data
def make_sample():
    rng = np.random.default_rng(21)
    n = 200
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.20, 0.55])
    sex = rng.choice(["male", "female"], n, p=[0.64, 0.36])
    age = np.clip(rng.gamma(4, 7.5, n) + 1, 1, 80).round(0)
    fare = (rng.gamma(2, 1, n) * np.select([pclass == 1, pclass == 2], [45, 20], 8)).round(2)
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09])
    cabin = rng.choice(["A12", "B22", "C85", "D10", "E46", "F33"], n)
    surv_prob = np.where(sex == "female", 0.75, 0.2) + np.where(pclass == 1, 0.15, 0)
    survived = (rng.random(n) < surv_prob).astype(int)
    d = pd.DataFrame({"passenger_id": np.arange(1, n + 1), "survived": survived, "pclass": pclass,
                      "sex": sex, "age": age, "fare": fare, "embarked": embarked, "cabin": cabin})
    d.loc[rng.random(n) < 0.20, "age"] = np.nan
    d.loc[rng.choice(n, 4, replace=False), "fare"] = np.nan
    d.loc[rng.choice(n, 2, replace=False), "embarked"] = np.nan
    d.loc[(pclass != 1) & (rng.random(n) < 0.95), "cabin"] = np.nan
    d.loc[(pclass == 1) & (rng.random(n) < 0.20), "cabin"] = np.nan
    return d


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
try:
    df = pd.read_csv(uploaded) if uploaded is not None else make_sample()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

# ---------------- Original analysis ----------------
st.header("1. Original missing-value analysis")
missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().mean() * 100).round(1)}).sort_values("missing_percent", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Missing cells", int(df.isnull().sum().sum()))
c3.metric("Rows lost if dropna()", f"{(1 - len(df.dropna()) / len(df)) * 100:.0f}%")

left, right = st.columns(2)
left.dataframe(missing, use_container_width=True)
right.bar_chart(missing["missing_percent"])

# ---------------- Choose methods ----------------
st.header("2. Choose a treatment for each column")
cols_with_na = [c for c in df.columns if df[c].isnull().any()]
if not cols_with_na:
    st.success("This dataset has no missing values.")
    st.stop()

clean = df.copy()
choices = {}
for c in cols_with_na:
    numeric = pd.api.types.is_numeric_dtype(df[c])
    options = (["Median", "Mean", "Mode", "Constant 0", "Drop rows", "Drop column"] if numeric
               else ["Mode", "Constant 'Unknown'", "Drop rows", "Drop column"])
    pct = df[c].isnull().mean() * 100
    hint = "suggested: Drop column" if pct > 60 else ("suggested: Median" if numeric else "suggested: Mode")
    choices[c] = st.selectbox(f"{c}  ({pct:.1f}% missing, {hint})", options, key=f"m_{c}")

for c, m in choices.items():
    if m == "Drop column":
        clean = clean.drop(columns=[c])
    elif m == "Drop rows":
        clean = clean[clean[c].notna()]
    elif m == "Median":
        clean[c] = clean[c].fillna(df[c].median())
    elif m == "Mean":
        clean[c] = clean[c].fillna(df[c].mean())
    elif m == "Mode":
        clean[c] = clean[c].fillna(df[c].mode()[0])
    elif m == "Constant 0":
        clean[c] = clean[c].fillna(0)
    elif m == "Constant 'Unknown'":
        clean[c] = clean[c].fillna("Unknown")

# ---------------- Result ----------------
st.header("3. Cleaned dataset")
r1, r2, r3 = st.columns(3)
r1.metric("Rows kept", f"{len(clean)} of {len(df)}")
r2.metric("Missing cells left", int(clean.isnull().sum().sum()))
r3.metric("Columns", f"{clean.shape[1]} of {df.shape[1]}")
st.dataframe(clean.head(20), use_container_width=True)

st.download_button("⬇️ Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"),
                   file_name="cleaned_data.csv", mime="text/csv")

with st.expander("Why these methods?"):
    st.markdown("""
- **Median** for skewed numerical columns: extreme values do not pull it.
- **Mean** for roughly symmetric numerical columns.
- **Mode** for categorical columns: the most frequent value.
- **Drop column** when most of the values are missing.
- **Drop rows** only when few rows are affected. A blind `dropna()` can delete most of the data.
""")
