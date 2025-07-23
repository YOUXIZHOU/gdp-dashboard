import pandas as pd
from pathlib import Path

# --- Configuration ----------------------------------------------------------
# Adjust these paths if your files live elsewhere in the Colab filesystem.
DATA_PATH = Path('sample_data.csv')  # sample_data.csv must be in the same folder

# --- Marketing keyword dictionaries ----------------------------------------
dictionaries = {
    'urgency_marketing': {
        'limited', 'limited time', 'limited run', 'limited edition', 'order now',
        'last chance', 'hurry', 'while supplies last', "before they're gone",
        'selling out', 'selling fast', 'act now', "don't wait", 'today only',
        'expires soon', 'final hours', 'almost gone'
    },
    'exclusive_marketing': {
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
        'members only', 'vip', 'special access', 'invitation only',
        'premium', 'privileged', 'limited access', 'select customers',
        'insider', 'private sale', 'early access'
    }
}

# --- Helper -----------------------------------------------------------------
def classify_statement(text: str) -> list[str]:
    """Return a list of dictionary names whose keywords appear in *text*."""
    text_lower = text.lower()
    matched: list[str] = []
    for label, keywords in dictionaries.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(label)
    return matched

# --- Pipeline ---------------------------------------------------------------

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Classify each statement
    df['labels'] = df['Statement'].astype(str).apply(classify_statement)

    # 3. (Optional) One‑hot encode each category for easier filtering/analysis
    for label in dictionaries:
        df[label] = df['labels'].apply(lambda cats, lbl=label: lbl in cats)

    # 4. Save the enriched DataFrame
    output_path = DATA_PATH.with_name(DATA_PATH.stem + '_classified.csv')
    df.to_csv(output_path, index=False)

    # 5. Quick sanity check
    print('Classification complete. Preview:')
    print(df.head())


if __name__ == '__main__':
    main()
