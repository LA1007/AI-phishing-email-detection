# check_filtered.py
import pandas as pd
import re

def is_mostly_english(text):
    'Check if text is primarily English'
    if not isinstance(text, str):
        return False
    letters = len(re.findall(r'[a-zA-Z]', text))
    total = len(re.sub(r'\s', '', text))
    if total == 0:
        return False
    return letters / total > 0.7

# Load original data
df = pd.read_csv('data/Phishing_Email.csv')
df = df.dropna(subset=['Email Text'])

# Record filtered emails
filtered_by_empty = df[df['Email Text'].str.lower().str.contains('empty')]
filtered_by_length = df[~(df['Email Text'].str.lower().str.contains('empty')) & (df['Email Text'].str.len() <= 10)]
filtered_by_lang = df[
    ~df['Email Text'].str.lower().str.contains('empty') &
    (df['Email Text'].str.len() > 10) &
    (~df['Email Text'].apply(is_mostly_english))
]

print("="*60)
print("Filtered Email Analysis")
print("="*60)

print(f"\n1. Emails containing 'empty': {len(filtered_by_empty)}")
print("   Examples:")
for i, row in filtered_by_empty.head(3).iterrows():
    print(f"   - {repr(row['Email Text'][:100])}")

print(f"\n2. Short emails (length <= 10): {len(filtered_by_length)}")
print("   Examples:")
for i, row in filtered_by_length.head(5).iterrows():
    print(f"   - {repr(row['Email Text'][:100])}")

print(f"\n3. Non-English/gibberish emails: {len(filtered_by_lang)}")
print("   Examples:")
for i, row in filtered_by_lang.head(10).iterrows():
    print(f"   - {repr(row['Email Text'][:100])}")

# Save to file
with open('filtered_samples.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("Filtered Email Examples\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"\n1. Emails containing 'empty' ({len(filtered_by_empty)}):\n")
    f.write("-"*40 + "\n")
    for i, row in filtered_by_empty.head(10).iterrows():
        f.write(f"{row['Email Text'][:200]}\n\n")
    
    f.write(f"\n2. Short emails ({len(filtered_by_length)}):\n")
    f.write("-"*40 + "\n")
    for i, row in filtered_by_length.head(10).iterrows():
        f.write(f"{row['Email Text'][:200]}\n\n")
    
    f.write(f"\n3. Non-English/gibberish emails ({len(filtered_by_lang)}):\n")
    f.write("-"*40 + "\n")
    for i, row in filtered_by_lang.head(20).iterrows():
        f.write(f"{row['Email Text'][:200]}\n\n")

print("\nFull examples saved to: filtered_samples.txt")
