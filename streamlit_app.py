from io import StringIO
import json
import re

import pandas as pd
import streamlit as st

# ──────────────────────────────  Page configuration  ──────────────────────────────
st.set_page_config(
    page_title="Text Transformation App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS for centered title and compact padding
st.markdown(
    """
    <style>
        .block-container h1:first-child {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 1.25rem;
        }
        .block-container { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] h2 {
            font-size: 1.05rem; margin-bottom: .3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Text Transformation App")

# ──────────────────────────────  Default dictionary  ──────────────────────────────
DEFAULT_DICT = {
    "Fashion": ["style", "fashion", "wardrobe", "clothing", "outfit"],
    "Food": ["delicious", "food", "dinner", "lunch", "restaurant"],
    "Travel": ["travel", "trip", "vacation", "explore", "journey"],
    "Fitness": ["workout", "fitness", "exercise", "gym", "training"],
}

# ──────────────────────────────  Sidebar ──────────────────────────────

st.sidebar.markdown(
    """
    **🔗 Related Tools**  
    • [Dictionary Refinement](https://claude.ai/public/artifacts/d537c8ac-0fe2-49f9-80df-157b753fd783)  
    • [Classification](https://gdp-dashboard-54gvj7gbgcw.streamlit.app/)  
    • [Join table](https://claude.ai/public/artifacts/e6d1678e-eaaf-41db-8a18-16f89f778f36)  
    • [Word Metrics](https://claude.ai/public/artifacts/f4358b58-5f90-43f3-a8b2-43e0c6f1b0d7)  
    """,
    unsafe_allow_html=True,
)

## 1️⃣  File upload
st.sidebar.header("1️⃣  Upload your CSV")
uploaded_file = st.sidebar.file_uploader(
    "Choose an Instagram CSV file (≤200 MB)",
    type=["csv"],
    accept_multiple_files=False,
)

st.sidebar.markdown("---")

## 2️⃣  Keyword dictionary
st.sidebar.header("2️⃣  Modify keyword dictionary")

dict_input = st.sidebar.text_area(
    "Dictionary (JSON)",
    value=json.dumps(DEFAULT_DICT, indent=2),
    height=280,
)

try:
    keyword_dict = json.loads(dict_input)
    if not isinstance(keyword_dict, dict):
        raise ValueError("Dictionary root must be a JSON object (key → list).")
except (json.JSONDecodeError, ValueError) as e:
    st.sidebar.error(f"❌ {e}\nUsing default dictionary instead.")
    keyword_dict = DEFAULT_DICT

st.sidebar.markdown(
    """<small>🔧 Edit the JSON above to add/delete categories & keywords.</small>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────  Configurations  ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("3️⃣  Configurations")

use_context = st.sidebar.checkbox("Use rolling context window", value=True)

if use_context:
    window_size = st.sidebar.slider(
        "Window size (number of sentences before/after)",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        help="How many sentences before and after to include when classifying."
    )
else:
    window_size = 0

# CSV‑based column selection
include_hashtags = st.sidebar.checkbox("Treat hashtags as separate sentences", value=True)

# ──────────────────────────────  Helper functions  ──────────────────────────────

def classify_sentence_with_context(index: int, sentences: list[str], window_size: int, kw_dict: dict) -> str:
    start = max(0, index - window_size)
    end = min(len(sentences), index + window_size + 1)
    context_chunk = " ".join(sentences[start:end]).lower()
    for cat, kws in kw_dict.items():
        if any(k.lower() in context_chunk for k in kws):
            return cat
    return "Uncategorized"

def process_dataframe(df: pd.DataFrame, id_col: str, text_col: str, kw_dict: dict, window_size: int, include_hashtags: bool) -> pd.DataFrame:
    df = df.rename(columns={id_col: "ID", text_col: "Context"})
    
    optional_columns = []
    for col in ["Likes", "Comments"]:
        if col in df.columns:
            optional_columns.append(col)

    rows = []
    for _, row in df.iterrows():
        pattern = r"(?<=[.!?])\s+"
        if include_hashtags:
            pattern += r"|(?=#[^\s]+)"
        sentences = re.split(pattern, str(row["Context"]))
        cleaned = [re.sub(r"\s+", " ", s.strip()) for s in sentences]
        cleaned = [s for s in cleaned if s and not re.fullmatch(r"[.!?]+", s)]
        for i, s in enumerate(cleaned, start=1):
            category = classify_sentence_with_context(i - 1, cleaned, window_size, kw_dict)
            row_data = {
                "ID": row["ID"],
                "Context": row["Context"],
                "Sentence ID": i,
                "Statement": s,
                "Category": category,
            }
            for col in optional_columns:
                row_data[col] = row[col]
            rows.append(row_data)
    return pd.DataFrame(rows)



# ──────────────────────────────  How to Use  ──────────────────────────────

st.markdown(
    """
### How to Use
1. **Upload your CSV file** using the file uploader above  
2. **Select ID Column** – Choose the column that uniquely identifies each record  
3. **Select Context Column** – Choose the column containing the text to be transformed  
4. **Configure options** – Choose whether to include hashtags as separate sentences  
5. **Click _Transform_** – Process your data into sentence‑level format  
6. **Download results** – Get your transformed data as a CSV file  
    """
)

# ──────────────────────────────  Main logic  ──────────────────────────────
if uploaded_file is None:
    st.info("👈  Start by uploading a CSV file from the sidebar.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)

# Enable column selection once CSV is loaded
id_column = st.sidebar.selectbox("Select ID column", options=raw_df.columns.tolist(), index=0)
context_column = st.sidebar.selectbox("Select Context column", options=raw_df.columns.tolist(), index=1)

if st.sidebar.button("⚙️  Transform"):
    with st.spinner("Processing …"):
        final_df = process_dataframe(raw_df, id_column, context_column, keyword_dict, window_size, include_hashtags)

    st.success("Processing complete!")
    st.subheader("Preview of processed data")
    st.caption(f"Rolling context window: {'Enabled' if use_context else 'Disabled'}, size = {window_size}")
    st.dataframe(final_df, use_container_width=True)

    buff = StringIO()
    final_df.to_csv(buff, index=False)
    st.download_button(
        "💾  Download CSV",
        data=buff.getvalue(),
        mime="text/csv",
        file_name="transformed_text.csv",
    )
