"""
AutoInsight — Single-file automated EDA platform.
Upload a CSV, get instant metrics, interactive Plotly viz, and a Gemini-powered executive summary.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ── ENV & PAGE CONFIG ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AutoInsight", page_icon="📊", layout="wide")

# Initialize Gemini client (new google-genai SDK)
client = None
if API_KEY:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.sidebar.warning(f"Gemini init error: {e}")
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not found in .env. Executive Summary tab will be disabled.")


# ── HELPERS ───────────────────────────────────────────────────────
def extract_metadata(df: pd.DataFrame) -> dict:
    """Build a structured JSON metadata payload from a DataFrame."""
    metadata = {
        "dimensions": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": int(df.duplicated().sum())
        },
        "missing_values": {
            "total_missing": int(df.isnull().sum().sum()),
            "total_cells": int(df.size),
            "columns_with_missing": {
                str(k): int(v)
                for k, v in df.isnull().sum().items() if v > 0
            }
        }
    }

    # Numeric descriptors
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        metadata["numeric_features"] = {}
        for col in desc.index:
            row = desc.loc[col]
            metadata["numeric_features"][str(col)] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "25%": float(row["25%"]),
                "50%": float(row["50%"]),
                "75%": float(row["75%"]),
                "max": float(row["max"]),
                "skewness": float(numeric_df[col].skew())
            }

        # Pairwise correlations (top 5 positive / negative)
        if len(numeric_df.columns) >= 2:
            corr_mat = numeric_df.corr()
            pairs = []
            for i in range(len(corr_mat.columns)):
                for j in range(i + 1, len(corr_mat.columns)):
                    val = corr_mat.iloc[i, j]
                    if pd.notna(val):
                        pairs.append((corr_mat.columns[i], corr_mat.columns[j], float(val)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            pos = [p for p in pairs if p[2] > 0][:5]
            neg = [p for p in pairs if p[2] < 0][:5]
            metadata["correlations"] = {
                "top_positive": [
                    {"feature_1": a, "feature_2": b, "correlation": c} for a, b, c in pos
                ],
                "top_negative": [
                    {"feature_1": a, "feature_2": b, "correlation": c} for a, b, c in neg
                ]
            }
        else:
            metadata["correlations"] = "Insufficient numeric features for pairwise correlation."
    else:
        metadata["numeric_features"] = "No numeric features detected."
        metadata["correlations"] = "No numeric features detected."

    # Categorical / discrete summaries
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    metadata["categorical_features"] = {}
    for col in cat_df.columns:
        vc = df[col].value_counts(dropna=False)
        metadata["categorical_features"][str(col)] = {
            "unique_count": int(df[col].nunique()),
            "top_category": str(vc.index[0]) if len(vc) > 0 else None,
            "top_category_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0
        }

    return metadata


def build_llm_prompt(metadata_json: str) -> str:
    """Return the exact Principal Data Analyst persona prompt."""
    return f"""You are an expert Principal Data Analyst embedded in the AutoInsight automated EDA platform. Your mission is to analyze statistical metadata derived from an uploaded tabular dataset and produce an executive-ready, highly actionable analytical summary.

### Core Objectives
1. Translate raw statistical metrics (distributions, central tendencies, skews, correlations, missingness) into clear business, academic, or operational narratives.
2. Flag anomalies, risks, data hygiene issues, and standout patterns that require immediate human investigation.
3. Recommend specific next steps (e.g., transformations, domain-specific deep dives, modeling avenues).

### Input Specification
You will receive structured JSON metadata containing:
- High-level dimensions (row and column counts)
- Missing value summaries
- Descriptive statistics for numeric features (mean, std, min, 25%, 50%, 75%, max)
- Top positive and negative pairwise correlations
- High-cardinality flags or category counts for key discrete features

### Behavioral Constraints & Guardrails
- GROUNDED IN EVIDENCE: Only make assertions directly supported by the provided summary stats. Never fabricate raw row data, names, or unverified narratives.
- AVOID VAGUE DESCRIPTIONS: Do not say "X is somewhat correlated with Y." Say "X and Y exhibit a strong positive linear correlation (r = 0.82), suggesting..."
- NO STATISTICAL RESTATEMENT: Do not simply list numbers the user can already see in a table. Explain *what the numbers mean* in context.
- CONCISENESS OVER LENGTH: Keep the entire output tight, scannable, and impact-driven.

### Required Output Format
You must structure your response strictly using the following four markdown sections:

**1. Executive Summary**
A 2-sentence synthesis of the dataset's primary narrative and structural integrity.

**2. Key Drivers & Relationships**
- [Bullet 1]: Detail the strongest observed correlation or relationship and its plausible analytical implication.
- [Bullet 2]: Detail a secondary relationship, trend, or category concentration.

**3. Anomalies & Data Hygiene Flags**
- [Bullet 1]: Flag any severe skew, high variance (large gap between mean and median), unexpected min/max ranges, or missing data concentrations.

**4. Recommended Next Steps**
- Provide 2 distinct, actionable steps (e.g., feature engineering suggestions, specific bivariate plots to examine, or outlier treatment protocols).

### Dataset Metadata (JSON)
{metadata_json}

Generate the analysis now."""


# ── UI ────────────────────────────────────────────────────────────
st.title("📊 AutoInsight")
st.caption("Upload any CSV. Get instant metrics, interactive visualizations, and an AI-generated executive summary.")

uploaded_file = st.file_uploader("📁 Drop your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # ── METRIC CARDS ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{len(df):,}")
    m2.metric("Total Columns", len(df.columns))
    total_missing = int(df.isnull().sum().sum())
    pct_missing = (total_missing / df.size * 100) if df.size > 0 else 0.0
    m3.metric("Missing Values", f"{total_missing:,}", f"{pct_missing:.1f}%")
    dupes = int(df.duplicated().sum())
    m4.metric("Duplicate Rows", f"{dupes:,}", "Clean" if dupes == 0 else "Review")

    # ── TABS ──────────────────────────────────────────────────────
    tab_overview, tab_dist, tab_corr, tab_exec = st.tabs(
        ["📝 Overview", "📈 Distributions", "🔥 Correlations", "🧠 Executive Summary"]
    )

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────
    with tab_overview:
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Descriptive Statistics")
        desc = df.describe()
        if not desc.empty:
            st.dataframe(desc.T, use_container_width=True)
        else:
            st.info("No numeric columns available for descriptive statistics.")

    # ── TAB 2: DISTRIBUTIONS ────────────────────────────────────
    with tab_dist:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if numeric_cols:
            sel_num = st.selectbox("Select numeric feature", numeric_cols, key="num_sel")
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    df, x=sel_num, marginal="box",
                    title=f"Distribution of {sel_num}",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(
                    df, y=sel_num,
                    title=f"Box Plot of {sel_num}",
                    template="plotly_white"
                )
                st.plotly_chart(fig_box, use_container_width=True)

        if categorical_cols:
            st.divider()
            sel_cat = st.selectbox("Select categorical feature", categorical_cols, key="cat_sel")
            vc = df[sel_cat].value_counts().head(20).reset_index()
            vc.columns = [sel_cat, "count"]
            fig_bar = px.bar(
                vc, x=sel_cat, y="count",
                title=f"Top Categories in {sel_cat}",
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if not numeric_cols and not categorical_cols:
            st.info("No plottable columns detected.")

    # ── TAB 3: CORRELATIONS ─────────────────────────────────────
    with tab_corr:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Pairwise Pearson Correlation Heatmap",
                template="plotly_white"
            )
            fig_corr.update_layout(height=max(400, len(corr.columns) * 40))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns to render a correlation heatmap.")

    # ── TAB 4: EXECUTIVE SUMMARY (LLM) ──────────────────────────
    with tab_exec:
        if client is None:
            st.error("🔑 Gemini API key is missing. Add `GEMINI_API_KEY` to your `.env` file and restart.")
        else:
            if st.button("🚀 Generate Executive Summary", type="primary", use_container_width=True):
                with st.spinner("Principal Data Analyst is reviewing your dataset..."):
                    try:
                        metadata = extract_metadata(df)
                        meta_json = json.dumps(metadata, indent=2, default=str)
                        prompt = build_llm_prompt(meta_json)

                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.2,
                                max_output_tokens=2048
                            )
                        )

                        st.markdown("---")
                        st.markdown(response.text)
                        st.markdown("---")

                        with st.expander("🔍 View raw metadata sent to LLM"):
                            st.json(metadata)

                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
            else:
                st.info("Click the button above to generate the AI-powered executive summary.")