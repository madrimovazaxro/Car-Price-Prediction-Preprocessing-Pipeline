```markdown
# Car Price Prediction & Preprocessing Pipeline

An end-to-end Python-based data science workspace focused on the exploratory data analysis (EDA), cleaning, and comprehensive data preparation of the CarDekho used cars dataset. This pipeline applies dynamic dataset ingestion, automated feature profiling, statistical distribution charting, non-trivial multi-modal null imputation, and mathematical outlier removal to produce high-integrity analysis matrices optimized for predictive regressors.

## Project Structure & Architecture
The logic inside this repository covers the fundamental data curation stages of the standard CRISP-DM life cycle:
1. **Automated Source Acquisition**: Employs programmatic bindings (`opendatasets`) to fetch data assets straight from Kaggle, ensuring highly reproducible build environments.
2. **Structural Data Profiling**: Runs dimensional shape checks and evaluates feature type designations.
3. **Completeness Analysis**: Assesses systematic missingness across categorical/numerical feature vectors using explicit spatial distribution patterns.
4. **Multi-Modal Imputation Architecture**: Isolates and handles discrete data variables according to their dtype distributions (robust center-tendency values for structural continuous values, and static label definitions for text data elements).
5. **Statistical Outlier Detection**: Filters out market skewness and anomalies using the Interquartile Range (IQR) method on the dependent label distribution space.

---

## Dataset Layout

This workflow operates on the **CarPrice Prediction Dataset (CarDekho)** sourced dynamically via Kaggle's open platform API.

### Dataset Dimensions & Scale
* **Initial Profile Size**: 8,128 baseline car entries mapped across 12 feature variables.
* **Cleaned Feature Space**: Extracted matrices containing 7,309 high-confidence observation targets.

### Feature Catalog & Metadata
* `name`: Brand name and model designation text.
* `year`: Registration timeline year.
* `selling_price`: Transaction value (Dependent target classification vector).
* `km_driven`: Total odometer distance traveled.
* `fuel`: Propellant fuel category types (`Diesel`, `Petrol`, `CNG`, `LPG`).
* `seller_type`: Sales channel designation profile (`Individual`, `Dealer`, `Trustmark Dealer`).
* `transmission`: Gear shift configuration classification (`Manual`, `Automatic`).
* `owner`: Vehicle ownership string lineage position (`First Owner`, `Second Owner`, etc.).
* `mileage(km/ltr/kg)`: Standard fuel efficiency metric.
* `engine`: Engine displacement size measured in CC (cubic centimeters).
* `max_power`: Engine brake horsepower configuration.
* `seats`: Operational passenger capacity constraint allocation.

---

## Implementation Details

### Setup & Package Installation
To install the package dependencies required to run the workflow notebook, run:
```bash
pip install opendatasets pandas numpy matplotlib seaborn

```

### 1. Programmatic Data Download

To bypass environment restrictions and avoid manual local asset storage placement, data assets are pulled programmatically using an API pipeline wrapper:

```python
import opendatasets as od
import pandas as pd

# Automatically checks local cache or queries Kaggle authentication credentials
od.download("[https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset/data](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset/data)")

# Ingest underlying CSV structured text source
df = pd.read_csv("./car-price-prediction-dataset/cardekho.csv")

```

### 2. Missing Value Analysis

A quantitative audit of data completeness highlights overlapping rows with null entries among engine specifications, matching a uniform ~2.72% missing rate:

| Target Feature Vector Column | Raw Omissions Count | Missing Percentage |
| --- | --- | --- |
| `seats` | 221 | 2.72% |
| `mileage(km/ltr/kg)` | 221 | 2.72% |
| `engine` | 221 | 2.72% |
| `max_power` | 215 | 2.65% |

#### Visualizing Missingness Patterns

A specialized matrix heatmap exposes structural alignment, indicating the missing parameters likely occur across identical unrecorded vehicle profiles:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Map out null positions to look for systematic failure alignments
sns.heatmap(df.isnull(), cbar=True, cmap="Reds")
plt.show()

```

### 3. Multi-Modal Imputation Strategy

To protect dataset balance and maximize total remaining records without losing core structural information, a dual-track strategy fills missing values based on column feature types:

* **Categorical Imputation**: Replaces missing entries with an isolated text fallback classification label string (`'None'`).
* **Numerical Imputation**: Uses outlier-resilient central tendency metrics (`.median()`) to preserve data distributions against skew variance.

```python
# Isolate text features and apply static labels
cat_all = df.select_dtypes(include='object').columns
df[cat_all] = df[cat_all].fillna('None')

# Isolate numeric columns and impute with median centers
num = df.select_dtypes(include="number").columns
df[num] = df[num].fillna(df[num].median())

```

Alternatively, rows with missing engineering data can be cleaned by dropping missing vectors directly:

```python
df.dropna(inplace=True)

```

### 4. IQR Outlier Eviction

Extreme values on high-end luxury models can distort predictive pricing algorithms. The workflow sets up defensive boundaries utilizing Interquartile Range (IQR) calculations to clean the distribution space:

$$\text{IQR} = Q_3 - Q_1$$

$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$

$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

```python
Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1

# Define structural variance bounds matrix mask
mask = (df["selling_price"] >= Q1 - 1.5 * IQR) & (df['selling_price'] <= Q3 + 1.5 * IQR)
df_clean = df[mask]

```

#### Visualizing Distribution Changes

Validating outlier removal across distribution arrays via side-by-side boxplots:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot raw distribution
sns.boxplot(data=df, y='selling_price', ax=axes[0])
axes[0].set_title('Distribution with Extreme Outliers')

# Plot normalized clean matrix structure 
sns.boxplot(data=df_clean, y='selling_price', ax=axes[1])
axes[1].set_title("Cleaned Distribution (Outliers Evicted)")

plt.tight_layout()
plt.show()

```

---

## Future Enhancements Roadmap

* [ ] Encode category arrays (`fuel`, `transmission`, `seller_type`, `owner`) using One-Hot encoding wrappers.
* [ ] Implement feature scaling transformations (e.g., standard normalization) on high-variance metrics like `km_driven`.
* [ ] Split clean information matrices into distinct train-test subsets for validation splits.
* [ ] Baseline performance testing across advanced modeling architectures (e.g., Random Forest Regressor, LightGBM, Gradient Boosting).

---

*Note: This documentation was compiled based on the exploratory engineering experiments conducted in the source notebook repository.*

```

```
