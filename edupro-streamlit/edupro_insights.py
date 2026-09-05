"""
EduPro Learner Insights — standalone Streamlit app

Run:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run edupro_insights.py

This file intentionally contains the complete app, including deterministic demo
data. Uploaded CSV files are parsed locally in the browser/session and are not
sent to a remote service by this app.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="EduPro Learner Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


USER_COLUMNS = ["UserID", "UserName", "Age", "Gender"]
COURSE_COLUMNS = [
    "CourseID",
    "CourseName",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
]
TRANSACTION_COLUMNS = ["TransactionID", "UserID", "CourseID", "TransactionDate"]
ALL_FILTERS = {
    "Age band": "All ages",
    "Gender": "All genders",
    "Category": "All categories",
    "Course type": "All course types",
    "Level": "All levels",
}


@st.cache_data
def demo_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a small deterministic dataset so the app is useful immediately."""
    users = pd.DataFrame(
        [
            ["U001", "Aisha Patel", 19, "Female"],
            ["U002", "Marcus Lee", 22, "Male"],
            ["U003", "Sofia Garcia", 27, "Female"],
            ["U004", "Daniel Okafor", 34, "Male"],
            ["U005", "Mei Chen", 41, "Female"],
            ["U006", "Noah Williams", 17, "Male"],
            ["U007", "Priya Shah", 29, "Female"],
            ["U008", "Ethan Brown", 38, "Male"],
            ["U009", "Lina Haddad", 24, "Female"],
            ["U010", "James Wilson", 46, "Male"],
            ["U011", "Olivia Martin", 52, "Female"],
            ["U012", "Ravi Kumar", 31, "Male"],
            ["U013", "Zoe Thompson", 20, "Female"],
            ["U014", "Amara Johnson", 36, "Female"],
            ["U015", "Mateo Rossi", 26, "Male"],
            ["U016", "Nina Petrova", 44, "Female"],
            ["U017", "Sam Taylor", 23, "Non-binary"],
            ["U018", "Grace Kim", 33, "Female"],
        ],
        columns=USER_COLUMNS,
    )
    courses = pd.DataFrame(
        [
            ["C001", "Foundations of Data", "Data & Technology", "Self-paced", "Beginner"],
            ["C002", "Python for Everyone", "Data & Technology", "Self-paced", "Beginner"],
            ["C003", "Product Design Studio", "Design & Creativity", "Instructor-led", "Intermediate"],
            ["C004", "Visual Storytelling", "Design & Creativity", "Self-paced", "Beginner"],
            ["C005", "Project Leadership", "Business", "Instructor-led", "Intermediate"],
            ["C006", "Strategic Thinking", "Business", "Self-paced", "Advanced"],
            ["C007", "Academic Writing", "Communication", "Self-paced", "Beginner"],
            ["C008", "Public Speaking Lab", "Communication", "Instructor-led", "Intermediate"],
            ["C009", "Mindful Productivity", "Personal Development", "Self-paced", "Beginner"],
            ["C010", "Leading with Empathy", "Personal Development", "Instructor-led", "Advanced"],
            ["C011", "Intro to Climate Action", "Sustainability", "Self-paced", "Beginner"],
            ["C012", "Community Health", "Health & Wellbeing", "Instructor-led", "Intermediate"],
        ],
        columns=COURSE_COLUMNS,
    )
    pairs = [
        ("U001", "C001"), ("U001", "C002"), ("U001", "C007"),
        ("U002", "C001"), ("U002", "C005"), ("U002", "C009"),
        ("U003", "C003"), ("U003", "C004"), ("U003", "C008"),
        ("U004", "C005"), ("U004", "C006"), ("U004", "C002"),
        ("U005", "C006"), ("U005", "C010"), ("U005", "C012"),
        ("U006", "C001"), ("U006", "C004"),
        ("U007", "C003"), ("U007", "C005"), ("U007", "C010"), ("U007", "C012"),
        ("U008", "C006"), ("U008", "C010"), ("U008", "C011"),
        ("U009", "C002"), ("U009", "C007"), ("U009", "C009"),
        ("U010", "C006"), ("U010", "C011"), ("U010", "C012"),
        ("U011", "C009"), ("U011", "C010"),
        ("U012", "C002"), ("U012", "C005"), ("U012", "C006"),
        ("U013", "C001"), ("U013", "C004"), ("U013", "C007"),
        ("U014", "C003"), ("U014", "C008"), ("U014", "C010"),
        ("U015", "C001"), ("U015", "C002"), ("U015", "C005"),
        ("U016", "C006"), ("U016", "C010"), ("U016", "C012"),
        ("U017", "C004"), ("U017", "C007"), ("U017", "C009"),
        ("U018", "C003"), ("U018", "C008"), ("U018", "C011"), ("U018", "C012"),
    ]
    transactions = pd.DataFrame(
        [
            [f"T{index:03d}", user_id, course_id, f"2025-{(index - 1) % 9 + 1:02d}-{(index - 1) % 26 + 1:02d}"]
            for index, (user_id, course_id) in enumerate(pairs, start=1)
        ],
        columns=TRANSACTION_COLUMNS,
    )
    return users, courses, transactions


def age_band(age: float) -> str:
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18–24"
    if age <= 34:
        return "25–34"
    if age <= 44:
        return "35–44"
    if age <= 54:
        return "45–54"
    return "55+"


def normalize_csv(uploaded_file: Any, columns: list[str], label: str) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        raw = pd.read_csv(uploaded_file)
        raw.columns = [str(column).strip() for column in raw.columns]
        lookup = {column.lower(): column for column in raw.columns}
        missing = [column for column in columns if column.lower() not in lookup]
        if missing:
            st.error(f"{label} is missing required columns: {', '.join(missing)}")
            return None
        result = raw[[lookup[column.lower()] for column in columns]].copy()
        result.columns = columns
        if "Age" in result:
            result["Age"] = pd.to_numeric(result["Age"], errors="coerce")
            result = result.dropna(subset=["Age"])
            result["Age"] = result["Age"].astype(int)
        return result.fillna("")
    except Exception as error:
        st.error(f"Could not read {label}: {error}")
        return None


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def filtered_dataset(
    users: pd.DataFrame,
    courses: pd.DataFrame,
    transactions: pd.DataFrame,
    selected: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    users = users.copy()
    courses = courses.copy()
    transactions = transactions.copy()
    users["AgeBand"] = users["Age"].map(age_band)
    joined = transactions.merge(users, on="UserID", how="inner", suffixes=("", "_user"))
    joined = joined.merge(courses, on="CourseID", how="inner", suffixes=("", "_course"))
    quality = {
        "unmatched_users": int(len(transactions) - len(transactions.merge(users[["UserID"]], on="UserID", how="inner"))),
        "unmatched_courses": int(len(transactions) - len(transactions.merge(courses[["CourseID"]], on="CourseID", how="inner"))),
    }
    if selected["Age band"] != "All ages":
        joined = joined[joined["AgeBand"] == selected["Age band"]]
    if selected["Gender"] != "All genders":
        joined = joined[joined["Gender"] == selected["Gender"]]
    if selected["Category"] != "All categories":
        joined = joined[joined["CourseCategory"] == selected["Category"]]
    if selected["Course type"] != "All course types":
        joined = joined[joined["CourseType"] == selected["Course type"]]
    if selected["Level"] != "All levels":
        joined = joined[joined["CourseLevel"] == selected["Level"]]
    joined["TransactionDate"] = pd.to_datetime(joined["TransactionDate"], errors="coerce")
    return joined, quality


def grouped_counts(frame: pd.DataFrame, column: str, value_name: str = "Enrollments") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column, value_name])
    return (
        frame.groupby(column, dropna=False)
        .size()
        .reset_index(name=value_name)
        .sort_values(value_name, ascending=False)
    )


def profile_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Build an automatic per-column EDA profile."""
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        date_like = (
            1.0
            if pd.api.types.is_datetime64_any_dtype(series)
            else pd.to_datetime(series, errors="coerce").notna().mean()
            if series.dtype == "object"
            else 0
        )
        outliers = 0
        if numeric.notna().sum() >= 4:
            q1, q3 = numeric.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                outliers = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
        rows.append(
            {
                "Dataset": name,
                "Column": column,
                "Data type": str(series.dtype),
                "Rows": len(series),
                "Missing": int(series.isna().sum()),
                "Missing %": round(float(series.isna().mean() * 100), 2),
                "Unique": int(series.nunique(dropna=True)),
                "Detected role": (
                    "date"
                    if date_like >= 0.8
                    else "numeric"
                    if numeric.notna().mean() >= 0.8
                    else "categorical/text"
                ),
                "Outliers": outliers,
            }
        )
    return pd.DataFrame(rows)


def dataset_quality(frame: pd.DataFrame) -> dict[str, int]:
    """Return automatic quality metrics used by the EDA panel."""
    numeric = frame.select_dtypes(include=np.number)
    outlier_cells = 0
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) >= 4:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                outlier_cells += int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "numeric_columns": len(numeric.columns),
        "categorical_columns": len(frame.select_dtypes(include=["object", "category"]).columns),
        "outlier_cells": outlier_cells,
    }


def engineer_learner_features(
    users: pd.DataFrame,
    courses: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically create reusable behavioral features from the three EduPro tables."""
    users_clean = users.copy()
    courses_clean = courses.copy()
    transactions_clean = transactions.copy()
    users_clean["Age"] = pd.to_numeric(users_clean["Age"], errors="coerce")
    users_clean["AgeBand"] = users_clean["Age"].map(age_band)
    transactions_clean["TransactionDate"] = pd.to_datetime(
        transactions_clean["TransactionDate"], errors="coerce"
    )
    joined = transactions_clean.merge(users_clean, on="UserID", how="inner")
    joined = joined.merge(courses_clean, on="CourseID", how="inner")
    if joined.empty:
        return pd.DataFrame(), joined

    joined["EnrollYear"] = joined["TransactionDate"].dt.year
    joined["EnrollMonth"] = joined["TransactionDate"].dt.month
    joined["EnrollQuarter"] = joined["TransactionDate"].dt.quarter
    joined["EnrollWeekday"] = joined["TransactionDate"].dt.day_name()

    features = (
        joined.groupby("UserID")
        .agg(
            Learner=("UserName", "first"),
            Age=("Age", "first"),
            Gender=("Gender", "first"),
            AgeBand=("AgeBand", "first"),
            TotalEnrollments=("TransactionID", "count"),
            DistinctCourses=("CourseID", "nunique"),
            DistinctCategories=("CourseCategory", "nunique"),
            DistinctCourseTypes=("CourseType", "nunique"),
            DistinctLevels=("CourseLevel", "nunique"),
            FirstEnrollment=("TransactionDate", "min"),
            LastEnrollment=("TransactionDate", "max"),
            ActiveDays=("TransactionDate", lambda values: (values.max() - values.min()).days),
        )
        .reset_index()
    )
    features["RepeatLearner"] = (features["TotalEnrollments"] > 1).astype(int)
    features["EnrollmentsPer30Days"] = (
        features["TotalEnrollments"] / features["ActiveDays"].clip(lower=30) * 30
    ).round(3)

    for column, prefix in [
        ("CourseCategory", "Category"),
        ("CourseType", "Type"),
        ("CourseLevel", "Level"),
        ("EnrollMonth", "Month"),
    ]:
        counts = pd.crosstab(joined["UserID"], joined[column]).add_prefix(f"{prefix}_")
        features = features.merge(counts, on="UserID", how="left")

    numeric_columns = features.select_dtypes(include=np.number).columns
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    features[numeric_columns] = features[numeric_columns].fillna(0)
    return features, joined


def run_ml_pipeline(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Automatically select KMeans k, add PCA coordinates, and flag anomalies."""
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    excluded = {
        "UserID",
        "Learner",
        "Gender",
        "AgeBand",
        "FirstEnrollment",
        "LastEnrollment",
    }
    model_columns = [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]
    matrix = features[model_columns].copy().replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    if len(matrix) < 4 or len(model_columns) < 2:
        return pd.DataFrame(), pd.DataFrame(), None

    scaled = StandardScaler().fit_transform(matrix)
    max_k = min(8, len(matrix) - 1)
    evaluations: list[dict[str, float | int]] = []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
        if len(set(labels)) > 1:
            evaluations.append(
                {
                    "Clusters": k,
                    "Silhouette score": round(float(silhouette_score(scaled, labels)), 4),
                }
            )
    evaluation_frame = pd.DataFrame(evaluations)
    if evaluation_frame.empty:
        return pd.DataFrame(), evaluation_frame, None
    best = evaluation_frame.sort_values("Silhouette score", ascending=False).iloc[0]
    best_k = int(best["Clusters"])
    final_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = final_model.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled)
    anomaly_model = IsolationForest(random_state=42, contamination="auto")
    anomaly_labels = anomaly_model.fit_predict(scaled)
    output = features.copy()
    output["Segment"] = [f"Segment {label + 1}" for label in labels]
    output["Anomaly"] = np.where(anomaly_labels == -1, "Review", "Typical")
    output["PCA1"] = points[:, 0]
    output["PCA2"] = points[:, 1]
    return output, evaluation_frame, float(best["Silhouette score"])


def main() -> None:
    st.title("EduPro Learner Insights")
    st.caption("A practical, privacy-aware view of who learners are and what they choose to learn.")

    if "dataset" not in st.session_state:
        st.session_state.dataset = demo_dataset()
        st.session_state.source = "Demo dataset"

    demo_users, demo_courses, demo_transactions = demo_dataset()

    with st.sidebar:
        st.header("Data & filters")
        st.caption("Uploaded files stay in this Streamlit session.")
        use_demo = st.button("Reset to demo dataset", use_container_width=True)
        if use_demo:
            st.session_state.dataset = (demo_users, demo_courses, demo_transactions)
            st.session_state.source = "Demo dataset"
            st.rerun()

        st.subheader("Upload your CSVs")
        users_file = st.file_uploader("Users.csv", type=["csv"], key="users-upload")
        courses_file = st.file_uploader("Courses.csv", type=["csv"], key="courses-upload")
        transactions_file = st.file_uploader(
            "Transactions.csv", type=["csv"], key="transactions-upload"
        )
        uploaded = [
            normalize_csv(users_file, USER_COLUMNS, "Users.csv"),
            normalize_csv(courses_file, COURSE_COLUMNS, "Courses.csv"),
            normalize_csv(transactions_file, TRANSACTION_COLUMNS, "Transactions.csv"),
        ]
        if any(frame is not None for frame in uploaded):
            current = st.session_state.dataset
            st.session_state.dataset = tuple(
                uploaded[index] if uploaded[index] is not None else current[index]
                for index in range(3)
            )
            st.session_state.source = "Local CSV workspace"

    users, courses, transactions = st.session_state.dataset
    users = users.copy()
    courses = courses.copy()
    transactions = transactions.copy()

    options = {
        "Age band": ["All ages"] + sorted(users["Age"].map(age_band).unique().tolist()),
        "Gender": ["All genders"] + sorted(users["Gender"].astype(str).unique().tolist()),
        "Category": ["All categories"] + sorted(courses["CourseCategory"].astype(str).unique().tolist()),
        "Course type": ["All course types"] + sorted(courses["CourseType"].astype(str).unique().tolist()),
        "Level": ["All levels"] + sorted(courses["CourseLevel"].astype(str).unique().tolist()),
    }
    filter_columns = st.columns(5)
    selected = {
        name: filter_columns[index].selectbox(name, values)
        for index, (name, values) in enumerate(options.items())
    }
    joined, quality = filtered_dataset(users, courses, transactions, selected)
    engineered_features, full_joined = engineer_learner_features(
        users, courses, transactions
    )
    ml_output, cluster_evaluation, best_silhouette = run_ml_pipeline(engineered_features)
    profile_sources = {
        "Users": users,
        "Courses": courses,
        "Transactions": transactions,
        "Joined enrollments": full_joined,
        "Engineered learner features": engineered_features,
    }
    active_users = joined["UserID"].nunique()
    active_courses = joined["CourseID"].nunique()

    st.info(
        f"**{st.session_state.source}** · {len(users):,} learners · "
        f"{len(courses):,} courses · {len(transactions):,} source enrollments"
    )
    if quality["unmatched_users"] or quality["unmatched_courses"]:
        st.warning(
            f"{quality['unmatched_users']} enrollment(s) could not be matched to a learner and "
            f"{quality['unmatched_courses']} could not be matched to a course. "
            "Unmatched rows are excluded from analytics."
        )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Learners in view", format_number(active_users))
    metric_columns[1].metric("Enrollments", format_number(len(joined)))
    metric_columns[2].metric("Courses chosen", format_number(active_courses))
    metric_columns[3].metric(
        "Average enrollments",
        f"{len(joined) / active_users:.1f}" if active_users else "0.0",
    )

    overview, demographics, detail, auto_eda, feature_engineering, segments, setup = st.tabs(
        [
            "Overview",
            "Demographics",
            "Enrollment detail",
            "Automated EDA",
            "Feature engineering",
            "ML segments",
            "Data setup",
        ]
    )

    with overview:
        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("When learners join")
            by_month = joined.dropna(subset=["TransactionDate"]).assign(
                Month=lambda frame: frame["TransactionDate"].dt.to_period("M").astype(str)
            )
            monthly = grouped_counts(by_month, "Month")
            if not monthly.empty:
                chart = px.area(
                    monthly,
                    x="Month",
                    y="Enrollments",
                    markers=True,
                    title="Enrollment rhythm",
                    color_discrete_sequence=["#3454c5"],
                )
                st.plotly_chart(chart, use_container_width=True)
                st.download_button(
                    "Download chart data",
                    csv_bytes(monthly),
                    "enrollment-rhythm.csv",
                    "text/csv",
                )
            else:
                st.info("No dated enrollments are available for this view.")
        with right:
            st.subheader("What learners choose")
            categories = grouped_counts(joined, "CourseCategory")
            if not categories.empty:
                chart = px.pie(
                    categories,
                    names="CourseCategory",
                    values="Enrollments",
                    hole=0.55,
                    title="Category popularity",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(chart, use_container_width=True)
                st.download_button(
                    "Download chart data",
                    csv_bytes(categories),
                    "course-categories.csv",
                    "text/csv",
                )
        st.subheader("Course level preference")
        levels = grouped_counts(joined, "CourseLevel")
        if not levels.empty:
            chart = px.bar(
                levels,
                x="CourseLevel",
                y="Enrollments",
                text_auto=True,
                title="Beginner, intermediate, and advanced enrollment",
                color="CourseLevel",
                color_discrete_sequence=["#3f8f87", "#ec9b45", "#805a9e"],
            )
            st.plotly_chart(chart, use_container_width=True)
            st.download_button(
                "Download chart data",
                csv_bytes(levels),
                "level-preference.csv",
                "text/csv",
            )

    with demographics:
        age_counts = (
            pd.DataFrame({"AgeBand": ["Under 18", "18–24", "25–34", "35–44", "45–54", "55+"]})
            .merge(
                users.assign(AgeBand=users["Age"].map(age_band))
                .groupby("AgeBand")
                .size()
                .reset_index(name="Learners"),
                on="AgeBand",
                how="left",
            )
            .fillna(0)
        )
        age_counts["Learners"] = age_counts["Learners"].astype(int)
        left, right = st.columns(2)
        with left:
            st.subheader("Age bands")
            chart = px.bar(
                age_counts,
                x="AgeBand",
                y="Learners",
                text_auto=True,
                color_discrete_sequence=["#3f8f87"],
            )
            st.plotly_chart(chart, use_container_width=True)
            st.download_button(
                "Download age data",
                csv_bytes(age_counts),
                "age-bands.csv",
                "text/csv",
            )
        with right:
            st.subheader("Gender mix")
            gender_counts = users["Gender"].value_counts().rename_axis("Gender").reset_index(name="Learners")
            chart = px.pie(
                gender_counts,
                names="Gender",
                values="Learners",
                hole=0.55,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(chart, use_container_width=True)
            st.download_button(
                "Download gender data",
                csv_bytes(gender_counts),
                "gender-mix.csv",
                "text/csv",
            )
        st.subheader("Plain-language readout")
        if not joined.empty:
            top_category = grouped_counts(joined, "CourseCategory").iloc[0]
            top_level = grouped_counts(joined, "CourseLevel").iloc[0]
            st.success(
                f"{top_category['CourseCategory']} is the most selected category with "
                f"{int(top_category['Enrollments'])} enrollments. "
                f"{top_level['CourseLevel']} courses lead the current filtered view."
            )
        else:
            st.info("Widen the filters to generate a learner readout.")

    with detail:
        st.subheader("Enrollment records")
        if joined.empty:
            st.info("No enrollment records match the selected filters.")
        else:
            detail_columns = [
                "TransactionID",
                "UserName",
                "Age",
                "Gender",
                "CourseName",
                "CourseCategory",
                "CourseType",
                "CourseLevel",
                "TransactionDate",
            ]
            detail_frame = joined[detail_columns].sort_values(
                "TransactionDate", ascending=False
            )
            search = st.text_input("Search learner, course, category, or level")
            if search:
                mask = detail_frame.astype(str).apply(
                    lambda column: column.str.contains(search, case=False, na=False)
                ).any(axis=1)
                detail_frame = detail_frame[mask]
            st.dataframe(detail_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered enrollments",
                csv_bytes(detail_frame),
                "edupro-enrollment-records.csv",
                "text/csv",
            )

    with auto_eda:
        st.subheader("Automated exploratory data analysis")
        st.caption(
            "Every upload is profiled automatically for shape, types, missing values, "
            "duplicates, unique values, date-like fields, and IQR outliers."
        )
        source_name = st.selectbox("Dataset to inspect", list(profile_sources))
        source_frame = profile_sources[source_name]
        quality_metrics = dataset_quality(source_frame)
        quality_columns = st.columns(7)
        quality_columns[0].metric("Rows", f"{quality_metrics['rows']:,}")
        quality_columns[1].metric("Columns", f"{quality_metrics['columns']:,}")
        quality_columns[2].metric("Duplicates", f"{quality_metrics['duplicate_rows']:,}")
        quality_columns[3].metric("Missing cells", f"{quality_metrics['missing_cells']:,}")
        quality_columns[4].metric("Numeric", f"{quality_metrics['numeric_columns']:,}")
        quality_columns[5].metric("Categorical", f"{quality_metrics['categorical_columns']:,}")
        quality_columns[6].metric("Outlier cells", f"{quality_metrics['outlier_cells']:,}")

        eda_profile = profile_frame(source_frame, source_name)
        st.dataframe(eda_profile, use_container_width=True, hide_index=True)
        st.download_button(
            "Download EDA profile",
            csv_bytes(eda_profile),
            "edupro-eda-profile.csv",
            "text/csv",
        )

        numeric_columns = source_frame.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = source_frame.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if numeric_columns:
            selected_numeric = st.selectbox("Numeric distribution", numeric_columns)
            numeric_chart = px.histogram(
                source_frame,
                x=selected_numeric,
                nbins=20,
                title=f"Distribution of {selected_numeric}",
                color_discrete_sequence=["#3454c5"],
            )
            st.plotly_chart(numeric_chart, use_container_width=True)
        if categorical_columns:
            selected_categorical = st.selectbox(
                "Categorical distribution", categorical_columns
            )
            category_values = (
                source_frame[selected_categorical]
                .astype(str)
                .value_counts()
                .head(20)
                .rename_axis(selected_categorical)
                .reset_index(name="Count")
            )
            category_chart = px.bar(
                category_values,
                x=selected_categorical,
                y="Count",
                title=f"Top values in {selected_categorical}",
                color_discrete_sequence=["#3f8f87"],
            )
            st.plotly_chart(category_chart, use_container_width=True)

    with feature_engineering:
        st.subheader("Automatically engineered learner features")
        st.caption(
            "Features are generated from joins, dates, enrollment behavior, course diversity, "
            "repeat activity, and one-hot category/type/level counts."
        )
        if engineered_features.empty:
            st.warning("No valid joined enrollments are available for feature engineering.")
        else:
            feature_columns = [
                column
                for column in engineered_features.columns
                if column not in {"UserID", "Learner", "Gender", "AgeBand"}
            ]
            feature_metrics = st.columns(4)
            feature_metrics[0].metric("Learners engineered", f"{len(engineered_features):,}")
            feature_metrics[1].metric("Generated features", f"{len(feature_columns):,}")
            feature_metrics[2].metric(
                "Repeat learners",
                f"{int(engineered_features['RepeatLearner'].sum()):,}",
            )
            feature_metrics[3].metric(
                "Avg. course diversity",
                f"{engineered_features['DistinctCourses'].mean():.1f}",
            )
            st.dataframe(engineered_features, use_container_width=True, hide_index=True)
            st.download_button(
                "Download engineered features",
                csv_bytes(engineered_features),
                "edupro-engineered-features.csv",
                "text/csv",
            )
            st.markdown(
                """
                **Automatic transformations applied**
                - Age bands and learner demographic normalization
                - Enrollment year, month, quarter, and weekday
                - Total enrollments, distinct courses, categories, types, and levels
                - First/last enrollment dates, active span, and enrollment rate
                - Repeat-learner indicator
                - Count features for every observed category, course type, level, and month
                - Missing and infinite numeric values replaced with safe numeric defaults
                """
            )

    with segments:
        st.subheader("Descriptive learner segments")
        st.caption(
            "The app automatically evaluates cluster counts from 2 to 8, selects the best "
            "silhouette score, then runs KMeans, PCA, and Isolation Forest. This is descriptive "
            "segmentation and anomaly screening, not prediction or automated decision-making."
        )
        if ml_output.empty:
            st.info("At least four learners with at least two observed preference features are needed.")
        else:
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Selected clusters",
                int(ml_output["Segment"].nunique()),
            )
            metric_columns[1].metric(
                "Best silhouette score",
                f"{best_silhouette:.2f}" if best_silhouette is not None else "n/a",
                help="Higher values indicate cleaner separation between segments.",
            )
            metric_columns[2].metric(
                "Anomalies flagged",
                int((ml_output["Anomaly"] == "Review").sum()),
            )
            chart = px.scatter(
                ml_output,
                x="PCA1",
                y="PCA2",
                color="Segment",
                symbol="Anomaly",
                hover_data=["Learner", "Age", "Gender", "TotalEnrollments", "Anomaly"],
                title="Learner preference map",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(chart, use_container_width=True)
            summary = (
                ml_output.groupby("Segment")
                .agg(
                    Learners=("UserID", "count"),
                    AverageAge=("Age", "mean"),
                    AverageEnrollments=("TotalEnrollments", "mean"),
                    ReviewFlags=("Anomaly", lambda values: (values == "Review").sum()),
                )
                .reset_index()
            )
            summary["AverageAge"] = summary["AverageAge"].round(1)
            summary["AverageEnrollments"] = summary["AverageEnrollments"].round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.subheader("Automatic cluster evaluation")
            st.dataframe(cluster_evaluation, use_container_width=True, hide_index=True)
            st.download_button(
                "Download segment assignments",
                csv_bytes(ml_output),
                "learner-segments.csv",
                "text/csv",
            )

    with setup:
        st.subheader("Data setup")
        st.write(
            "Provide three CSV files with the following column names. Files can be uploaded "
            "in any order. The app automatically joins them, profiles the data, engineers "
            "behavioral features, evaluates clusters, and generates downloadable outputs."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "File": ["Users.csv", "Courses.csv", "Transactions.csv"],
                    "Required columns": [
                        ", ".join(USER_COLUMNS),
                        ", ".join(COURSE_COLUMNS),
                        ", ".join(TRANSACTION_COLUMNS),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "Privacy note: this standalone app does not include authentication or a hosted database. "
            "Use a trusted deployment and review your organization’s data policy before uploading real learner records."
        )
        st.subheader("Current source")
        st.write(
            f"{st.session_state.source}: {len(users):,} learners, {len(courses):,} courses, "
            f"{len(transactions):,} enrollments."
        )


if __name__ == "__main__":
    main()