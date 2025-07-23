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

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Union

import pandas as pd
import streamlit as st

################################################################################
# 🔧 Default dictionaries -------------------------------------------------------
################################################################################

# NOTE: We store values as **lists** (NOT sets) so the object is JSON‑serialisable
DEFAULT_DICTIONARIES: Dict[str, List[str]] = {
    "greeting": [
        "hello",
        "hi",
        "good morning",
        "good afternoon",
        "good evening",
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "see you",
    ],
    "gratitude": [
        "thank you",
        "thanks",
        "much appreciated",
    ],
}

################################################################################
# 🏷️ Utility functions ---------------------------------------------------------
################################################################################

def compile_patterns(dic: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    """Pre‑compile regex OR‑patterns for each category."""
    patterns: Dict[str, re.Pattern] = {}
    for category, phrases in dic.items():
        escaped = [re.escape(p.strip()) for p in phrases if p.strip()]
        if not escaped:
            continue
        pattern = re.compile(r"(?:^|\b)(" + "|".join(escaped) + r")(?:$|\b)", re.IGNORECASE)
        patterns[category] = pattern
    return patterns


def classify(text: str, patterns: Dict[str, re.Pattern]) -> str:
    """Return the first category whose pattern matches *text*; else 'other'."""
    for category, pat in patterns.items():
        if pat.search(text):
            return category
    return "other"

################################################################################
# 🎈 Streamlit UI --------------------------------------------------------------
################################################################################

def main() -> None:
    st.set_page_config(page_title="Dictionary Classifier", page_icon="🗂️", layout="centered")
    st.title("🗂️ Dictionary‑Based Text Classifier")

    # Sidebar ‑‑ dictionary editor
    st.sidebar.header("🔧 Edit Dictionaries")
    st.sidebar.markdown("Enter/modify a JSON object where *keys* are category names and *values* are **lists of phrases**.")

    # Convert default dict to JSON string safely (lists are already JSON‑serialisable)
    dict_str = json.dumps(DEFAULT_DICTIONARIES, indent=2, ensure_ascii=False)
    dict_input = st.sidebar.text_area("Classification dictionary (JSON)", value=dict_str, height=300)

    # Validate user JSON
    try:
        user_dic_raw: Dict[str, Union[List[str], Set[str]]] = json.loads(dict_input)
        # Ensure all values are lists of strings
        user_dic: Dict[str, List[str]] = {}
        for k, v in user_dic_raw.items():
            if isinstance(v, (list, set, tuple)):
                user_dic[k] = [str(x) for x in v]
            else:
                user_dic[k] = [str(v)]
    except Exception as e:
        st.sidebar.error(f"❌ Invalid JSON: {e}")
        st.stop()

    patterns = compile_patterns(user_dic)

    # File uploader
    st.header("📤 Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    text_column = st.text_input("Name of the text column", value="Statement")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found in the uploaded file.")
            st.stop()

        # Classification
        st.header("🏷️ Classification Results")
        df["predicted_category"] = df[text_column].astype(str).apply(lambda x: classify(x, patterns))
        st.dataframe(df.head())

        # Download link
        csv_out = df.to_csv(index=False).encode("utf‑8")
        st.download_button("⬇️ Download classified CSV", csv_out, file_name="classified_output.csv", mime="text/csv")
    else:
        st.info("👆 Upload a CSV file to get started.")


if __name__ == "__main__":
    main()
