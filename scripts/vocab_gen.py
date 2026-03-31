import pandas as pd
import json
import re

def build_feynman_vocab(csv_path, output_path="vocab.json"):
    df = pd.read_csv(csv_path)
    vocab = ["[PAD]", "[BOS]", "[EOS]", "[PRED]"]
    vocab += list("0123456789.e-")
    var_columns = [f"v{i}_name" for i in range(1, 11)]
    unique_vars = set()
    for col in var_columns:
        if col in df.columns:
            unique_vars.update(df[col].dropna().unique().astype(str).tolist())
    unique_ops = set()
    op_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    
    valid_formulas = df['Formula'].dropna().astype(str).tolist()
    
    for formula in valid_formulas:
        words = op_pattern.findall(formula)
        for w in words:
            if w not in unique_vars:
                unique_ops.add(w)
    
    symbols = list("+-*/^()")
    
    full_list = vocab + sorted(list(unique_vars)) + sorted(list(unique_ops)) + symbols
    word_to_id = {word: i for i, word in enumerate(full_list)}
    
    with open(output_path, 'w') as f:
        json.dump(word_to_id, f, indent=4)
    
    print(f"Vocab size: {len(word_to_id)} saved to {output_path}")
    
build_feynman_vocab("../../FeynmanEquations.csv")