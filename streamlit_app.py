"""
Streamlit Dictionary‑Based Text Classifier 🎈

This app lets users:
1. Upload a CSV file containing a text column (default **Statement**).
2. Edit or extend the dictionary‑based classification rules (category → phrases).
3. Run classification right in the browser.
4. Download the enriched CSV.

Converted from the standalone script `dictionary_classifier.py` on 2025‑07‑22.
"""

from __future__ import annotations

import ast
import json
from io import StringIO
from typing import Any, Dict, List, Set

import pandas as pd
import streamlit as st

################################################################################
# 🗂️ Default classification dictionaries (feel free to extend in the sidebar) #
################################################################################
DEFAULT_DICTIONARIES: Dict[str, Set[str]] = {
    "urgency_marketing": {
        "limited",
        "limited time",
        "limited run",
        "limited edition",
        "order now",
        "last chance",
        "hurry",
        "while supplies last",
        "before they're gone",
        "selling out",
        "selling fast",
        "act now",
        "don't wait",
        "today only",
        "expires soon",
        "final hours",
        "almost gone",
    },
    "exclusive_marketing": {
        "exclusive",
        "exclusively",
        "exclusive offer",
        "exclusive deal",
        "members only",
        "vip",
        "special access",
        "invitation only",
        "premium",
        "privileged",
        "limited access",
        "select customers",
        "insider",
        "private sale",
        "early access",
    },
}

################################################################################
# 🔧 Utility functions                                                           #
################################################################################

def parse_dictionaries(input_text: str) -> Dict[str, Set[str]]:
    """Safely parse the user's dictionary (Python literal)."""
    if not input_text.strip():
        return DEFAULT_DICTIONARIES

    try:
        data: Any = ast.literal_eval(input_text)
        if not isinstance(data, dict):
            raise ValueError("Parsed object is not a dict.")
        parsed: Dict[str, Set[str]] = {k: set(v) for k, v in data.items()}
        return parsed
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Failed to parse dictionaries: {exc}")
        st.stop()


def has_phrase(text: str, phrases: Set[str]) -> bool:
    """Return **True** if any phrase appears in *text* (case‑insensitive)."""
    return any(p in text for p in phrases)


def classify(text: str, dictionaries: Dict[str, Set[str]]) -> List[str]:
    """Return all dictionary keys matched in *text*."""
    text = str(text).lower()
    return [cat for cat, phrases in dictionaries.items() if has_phrase(text, phrases)]

################################################################################
# 🚀 Streamlit app                                                              #
################################################################################

def main() -> None:  # noqa: D401 – imperative mood intentional
    st.set_page_config(
        page_title="Dictionary‑Based Text Classifier",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Dictionary‑Based Text Classifier")
    st.write(
        "Upload a CSV, edit the dictionaries in the sidebar, and click **Classify** to "
        "add category labels to your data."
    )

    # Sidebar – data input
    st.sidebar.header("1️⃣ Upload your dataset")
    uploaded_file = st.sidebar.file_uploader(
        "CSV file", type=["csv"], key="csv_uploader"
    )
    column_name = st.sidebar.text_input("Text column name", value="Statement")

    # Sidebar – dictionary editing
    st.sidebar.header("2️⃣ Edit classification dictionaries")
    dict_text = st.sidebar.text_area(
        "Python dict (category -> list / set of phrases)",
        value=json.dumps(DEFAULT_DICTIONARIES, indent=2),
        height=300,
    )
    dictionaries = parse_dictionaries(dict_text)

    # Main area
    if uploaded_file is None:
        st.info("👈 Upload a CSV file to get started.")
        return

    # Read CSV into DataFrame
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Could not read CSV: {exc}")
        return

    if column_name not in df.columns:
        st.error(f"❌ Column '{column_name}' not found in the uploaded CSV.")
        return

    st.write("### 🔍 Data preview")
    st.dataframe(df.head())

    if st.button("🚀 Classify", use_container_width=True):
        # Perform classification
        with st.spinner("Classifying..."):
            df["_text_lower"] = df[column_name].fillna("").str.lower()
            df["labels"] = df["_text_lower"].apply(lambda t: classify(t, dictionaries))

            # Optional one‑hot columns
            for cat in dictionaries:
                df[cat] = df["_text_lower"].apply(
                    lambda t, phrases=dictionaries[cat]: has_phrase(t, phrases)
                )

            df.drop(columns="_text_lower", inplace=True)

        st.success("✅ Classification complete!")
        st.write("### 📝 Classified data preview")
        st.dataframe(df.head())

        # Download section
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download classified CSV",
            data=csv_buffer.getvalue(),
            file_name="classified_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
