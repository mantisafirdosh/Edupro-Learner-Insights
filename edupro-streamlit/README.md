# EduPro Learner Insights — Streamlit Project

This is the complete Streamlit version of the EduPro learner analytics project.
It is intentionally self-contained and includes embedded demo data, so it can
run immediately without a database or API server.

## What the app does automatically

After the CSV files are uploaded, the app automatically:

- Profiles rows, columns, data types, missing values, duplicates, and unique values
- Detects numeric outliers using the IQR method
- Joins learners, courses, and transactions
- Creates age bands and enrollment date features
- Engineers learner behavior features
- Creates category, type, level, and monthly count features
- Normalizes ML features with `StandardScaler`
- Tests multiple KMeans cluster counts
- Selects the best cluster count using silhouette score
- Creates a PCA learner visualization
- Flags unusual learner profiles with Isolation Forest
- Provides charts and CSV downloads for the generated results

## Project files

```text
edupro-streamlit/
├── edupro_insights.py
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.10 or newer.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the app

```bash
streamlit run edupro_insights.py
```

If the `streamlit` command is not found:

```bash
python -m streamlit run edupro_insights.py
```

Streamlit will show a local URL, usually:

```text
http://localhost:8501
```

## Upload data

The app expects these three CSV files:

### Users.csv

```text
UserID,UserName,Age,Gender
```

### Courses.csv

```text
CourseID,CourseName,CourseCategory,CourseType,CourseLevel
```

### Transactions.csv

```text
TransactionID,UserID,CourseID,TransactionDate
```

The app starts with a built-in demo dataset. Uploading valid CSV files
automatically replaces the demo data for the current Streamlit session.

## Main sections

- **Overview** — enrollment trends, category popularity, and course levels
- **Demographics** — age bands, gender mix, and plain-language findings
- **Enrollment detail** — searchable records and filtered downloads
- **Automated EDA** — data-quality metrics, distributions, and EDA profile download
- **Feature engineering** — generated learner feature table and download
- **ML segments** — automatic clustering, PCA map, silhouette evaluation, and anomaly flags
- **Data setup** — required columns and privacy notes

Uploaded data is processed in the Streamlit session by this application. It is
not sent to an external API by the code in this project.
