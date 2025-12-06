# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
# 📊 **Task 1

This task focuses on performing an in-depth exploratory analysis of the motor insurance dataset to understand its structure, quality, relationships, and trends. The EDA provides insights into policy information, customer characteristics, vehicle details, financial metrics, and claim behaviour over time.

---

## ✅ **1. Data Loading and Initial Understanding**

* Imported the dataset and loaded it into a pandas DataFrame.
* Inspected the first few rows to understand the schema and value patterns.
* Identified key categories of attributes:

  * **Policy-related attributes** (UnderwrittenCoverID, PolicyID, TransactionMonth…)
  * **Client demographics** (Citizenship, LegalType, Gender, Language…)
  * **Location attributes** (Country, Province, PostalCode, Cresta zones…)
  * **Vehicle details** (Make, Model, Cylinders, Kilowatts, RegistrationYear…)
  * **Financial & plan attributes** (SumInsured, CalculatedPremiumPerTerm, TotalPremium…)
  * **Claims information** (TotalClaims, Claim frequency/severity)

---

## ✅ **2. Data Cleaning & Preparation**

* Checked column data types and **casted dates** (e.g., TransactionMonth) to proper datetime format.
* Converted numerical fields (e.g., TotalClaims, TotalPremium, CustomValueEstimate) from string to numeric using `pd.to_numeric()`.
* Handled zero values and missing values using:

  * `df.isna().sum()`
  * `df.eq(0).sum()`
* Standardized date columns and derived **YearMonth** for temporal analysis.
* Removed ID-based columns from visualizations where they add no analytical value.

---

## ✅ **3. Univariate Analysis**

### 📌 Numerical Variables

* Generated **histograms** for key numerical features such as:

  * TotalClaims
  * TotalPremium
  * CustomValueEstimate
  * SumInsured
  * CapitalOutstanding
* Assessed distribution shapes, skewness, and presence of extreme values.

### 📌 Categorical Variables

* Created **bar charts** for:

  * Vehicle types
  * Makes and models
  * Client marital status, banking information, legal type, etc.
* Identified dominant categories and unusual values.

---

## ✅ **4. Outlier Detection**

* Used the **IQR (Interquartile Range)** method to statistically identify outliers in major financial attributes.
* Detected large right-skewed distributions in:

  * TotalClaims
  * CustomValueEstimate
  * SumInsured
* Examined outlier rows to understand potential high-impact claims or data inconsistencies.

---

## ✅ **5. Temporal Trends Analysis**

* Converted date variables and grouped data by monthly periods.
* Analyzed:

  * **Claim frequency** → number of months where claims were reported
  * **Claim severity** → average claim amount per month
* Visualized time-series trends using line plots.
* Identified potential seasonal patterns or unusual spikes across the 18-month period.

---

## ✅ **6. Vehicle Risk Analysis**

* Grouped claims by **vehicle make** and **make + model** combinations.
* Identified:

  * Top vehicles associated with **highest claim amounts**
  * Vehicles/models with **lowest or zero claim activity**
* Highlighted brand/model risk patterns for actuarial and underwriting considerations.

---

# 🎯 **Outcome of Task 1**

This exploratory data analysis provides a deep understanding of:

* Data quality and structure
* Customer demographics and vehicle characteristics
* Financial behaviour (Premiums, SumInsured, Claims)
* Risk distribution and outliers
* Time-based behaviour of claim patterns
* High-risk vs low-risk vehicle categories


# **Task 2 – Data Version Control (DVC) Setup & Dataset Versioning**

## **📌 Overview**

Task-2 focused on integrating **Data Version Control (DVC)** into the project to manage large datasets efficiently without pushing them to GitHub. The task included setting up local storage for data, tracking datasets with DVC, and creating versioned dataset snapshots.

---

## **⚙️ Objectives**

* Install and configure DVC in the project
* Set up a local DVC remote storage
* Add datasets to DVC tracking
* Create versioned dataset states
* Commit DVC metadata (`.dvc` files) to Git for reproducibility

---

## **🔧 Steps Completed**

### **1. Installed DVC**

DVC was installed using pip:

```bash
pip install dvc
```

---

### **2. Initialized DVC**

Inside the project directory:

```bash
dvc init
```

This created the `.dvc/` directory, config files, and prepared the project for data tracking.

---

### **3. Set Up Local DVC Remote Storage**

A directory was created to store actual dataset versions:

```bash
mkdir /path/to/local/storage
```

The directory was added as the default DVC remote:

```bash
dvc remote add -d localstorage /path/to/local/storage
```

---

### **4. Added Data to DVC**

Datasets placed inside the project folder were tracked using:

```bash
dvc add data/your_dataset.csv
```

DVC generated a pointer file (`your_dataset.csv.dvc`) and stored the real dataset in the DVC cache.

---

### **5. Committed Metadata to Git**

The dataset itself was **not** pushed to Git.
Instead, only DVC metadata and `.gitignore` updates were committed:

```bash
git add data/your_dataset.csv.dvc .gitignore
git commit -m "Add dataset version 1 using DVC"
```

---

### **6. Created Additional Dataset Versions**

Whenever the dataset was updated, a new version was created by:

```bash
dvc add data/your_dataset.csv
git add data/your_dataset.csv.dvc
git commit -m "Update dataset - version X"
```

---

### **7. Pushed Data to Remote Storage**

The actual dataset versions were stored in the local DVC remote:

```bash
dvc push
```

---

## **📂 What This Achieves**

* No large data is stored on GitHub
* Dataset versions are tracked through Git commits
* You can restore any historical dataset using:

```bash
git checkout <commit>
dvc pull
```

* Perfect setup for reproducible machine learning workflows

---

## **📎 Next Steps**

* Automate DVC workflow using `dvc.yaml` pipelines
* Connect DVC to cloud storage (optional)
* Integrate dataset versioning into your ML experimentation workflow

