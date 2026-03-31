import pandas as pd
import json

CSV_PATH = "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/FeynmanAugmentPrefixEquations.csv"
OUTPUT_VOCAB = "./vocab_bfgs_prefix.json"

def rebuild_prefix_vocab():
    df = pd.read_csv(CSV_PATH).dropna(subset=['Prefix_Formula'])
    base_symbols = ["[PAD]", "[BOS]", "[EOS]", "[PRED]", "[UNK]"]
    all_tokens = set(base_symbols)
    
    for formula in df['Prefix_Formula'].astype(str):
        tokens = formula.split()
        all_tokens.update(tokens)
    sorted_tokens = sorted(list(all_tokens))
    word_to_id = {word: i for i, word in enumerate(sorted_tokens)}
    if word_to_id.get("[PAD]") != 0:
        current_zero = [w for w, i in word_to_id.items() if i == 0][0]
        word_to_id[current_zero], word_to_id["[PAD]"] = word_to_id["[PAD]"], 0

    with open(OUTPUT_VOCAB, 'w') as f:
        json.dump(word_to_id, f, indent=4)
        
    print(f"New Prefix Vocab saved to {OUTPUT_VOCAB}")
    print(f"Total tokens: {len(word_to_id)}")
    print(f"Sample tokens: {list(word_to_id.keys())[:10]}")

if __name__ == "__main__":
    rebuild_prefix_vocab()
