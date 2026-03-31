import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
import os
import numpy as np

ATOM_REGEX = r'id\([\w\.]+\)|const\(\[CONST\]\)|[\w\.]+\(|[(),]'

def build_vocab(csv_path, output_path="vocab.json"):
    df = pd.read_csv(csv_path).dropna(subset=['Formula'])
    base_symbols = ["[PAD]", "[BOS]", "[EOS]", "[PRED]"]
    all_tokens = set(base_symbols)

    for f in df['Formula'].astype(str):
        tokens = re.findall(ATOM_REGEX, f)
        all_tokens.update(tokens)

    full_vocab = sorted(list(all_tokens))
    word_to_id = {word: i for i, word in enumerate(full_vocab)}

    if "[PAD]" in word_to_id and word_to_id["[PAD]"] != 0:
        current_zero_word = [w for w, i in word_to_id.items() if i == 0][0]
        word_to_id[current_zero_word], word_to_id["[PAD]"] = word_to_id["[PAD]"], 0

    with open(output_path, 'w') as f:
        json.dump(word_to_id, f, indent=4)
    print(f"Functional Vocab built with {len(word_to_id)} tokens.")


class FeynmanDataset(Dataset):
    def __init__(self, csv_path, data_dir, vocab_path="vocab.json", n_points=100, max_vars=12):
        df = pd.read_csv(csv_path).dropna(subset=['Filename', 'Formula'])
        df['Filename'] = df['Filename'].astype(str)

        valid_rows = []
        for _, row in df.iterrows():
            fname = row['Filename']
            if fname and fname.lower() != 'nan':
                full_path = os.path.join(data_dir, fname)
                if os.path.exists(full_path):
                    valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows)
        self.data_dir = data_dir
        self.n_points = n_points
        self.max_vars = max_vars

        with open(vocab_path, 'r') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {int(i): w for w, i in self.word_to_id.items()}

    def tokenize_formula(self, formula):
        f_str = str(formula).replace(" ", "")
        tokens = re.findall(ATOM_REGEX, f_str)
        return [self.word_to_id[t] for t in tokens if t in self.word_to_id]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.data_dir, row['Filename'])

        try:
            df_points = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
            sampled = df_points.sample(n=self.n_points, replace=True).values
            cloud = np.zeros((self.n_points, self.max_vars + 1))
            num_cols = min(sampled.shape[1], self.max_vars + 1)
            raw_target = sampled[:, -1]
            normed_values = np.sign(sampled) * np.log1p(np.abs(sampled))
            cloud[:, :num_cols] = normed_values[:, :num_cols]

        except Exception:
            cloud = np.zeros((self.n_points, self.max_vars + 1))
            raw_target = np.zeros(self.n_points)

        raw_ids = self.tokenize_formula(row['Formula'])
        eq_tokens = [self.word_to_id["[BOS]"]] + raw_ids + [self.word_to_id["[EOS]"]]

        return {
            "data_cloud": torch.tensor(cloud, dtype=torch.float32),
            "eq_tokens": torch.tensor(eq_tokens, dtype=torch.long),
            "raw_target": torch.tensor(raw_target, dtype=torch.float32)
        }


def collate_fn(batch):
    clouds = torch.stack([item['data_cloud'] for item in batch])
    raw_targets = torch.stack([item['raw_target'] for item in batch])
    eq_tokens = [item['eq_tokens'] for item in batch]
    eq_padded = torch.nn.utils.rnn.pad_sequence(eq_tokens, batch_first=True, padding_value=0)

    return {
        "data_cloud": clouds,
        "eq_tokens": eq_padded,
        "raw_target": raw_targets
    }


class FeynmanPrefixDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_dir,
        vocab_path="vocab.json",
        base_points=100,
        max_points=1000,
        max_vars=12
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Filename', 'Prefix_Formula'])

        valid_rows = []
        for _, row in df.iterrows():
            if os.path.exists(os.path.join(data_dir, str(row['Filename']))):
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.data_dir = data_dir

        self.base_points = base_points
        self.max_points = max_points
        self.max_vars = max_vars
        self.var_order = [
            "x", "y", "z", "v", "t", "r", "p", "m",
            "theta", "sigma", "theta1", "x1"
        ]

        with open(vocab_path, 'r') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {int(i): w for w, i in self.word_to_id.items()}

    def tokenize_prefix(self, formula_str):
        tokens = str(formula_str).split()
        unk_id = self.word_to_id.get("[UNK]", 0)
        return [self.word_to_id.get(t, unk_id) for t in tokens]

    def get_n_points(self, n_vars):
        return min(self.max_points, self.base_points * (n_vars ** 2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.data_dir, row['Filename'])

        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            n_vars = data.shape[1] - 1
            n_points = min(self.get_n_points(n_vars), self.max_points)
            indices = np.random.choice(len(data), n_points, replace=True)
            sampled = data[indices]

            raw_cloud = np.zeros((self.max_points, self.max_vars + 1), dtype=np.float32)
            normed_cloud = np.zeros((self.max_points, self.max_vars + 1), dtype=np.float32)
            mask = np.zeros(self.max_points, dtype=np.float32)

            num_cols = min(sampled.shape[1], self.max_vars + 1)
            raw_cloud[:n_points, :num_cols] = sampled[:, :num_cols]
            normed_vals = np.sign(sampled) * np.log1p(np.abs(sampled))
            normed_cloud[:n_points, :num_cols] = normed_vals[:, :num_cols]
            mask[:n_points] = 1.0

        except Exception:
            raw_cloud = np.zeros((self.max_points, self.max_vars + 1), dtype=np.float32)
            normed_cloud = np.zeros((self.max_points, self.max_vars + 1), dtype=np.float32)
            mask = np.zeros(self.max_points, dtype=np.float32)
            n_vars = 0
        OPERATOR_SET = {
            "add", "mul", "sub", "div", "pow",
            "sin", "cos", "exp", "log", "sqrt", "neg", "abs",
            "tanh", "asin", "acos", "atan",
            "const", "pi", "e",
            "[BOS]", "[EOS]", "[PAD]", "[PRED]", "[UNK]",
        }

        prefix_str = str(row['Prefix_Formula'])
        tokens = prefix_str.split()
        real_var_names = []
        seen = set()
        for tok in tokens:
            if tok in OPERATOR_SET:
                continue
            if re.match(r'^-?[\d\.]+([eE][+-]?\d+)?$', tok):
                continue
            if tok not in seen:
                seen.add(tok)
                real_var_names.append(tok)
        positional_fallback = self.var_order[:n_vars]
        for j, fallback in enumerate(positional_fallback):
            if j >= len(real_var_names):
                real_var_names.append(fallback)

        prefix_ids = self.tokenize_prefix(row['Prefix_Formula'])
        eq_tokens = [self.word_to_id["[BOS]"]] + prefix_ids + [self.word_to_id["[EOS]"]]

        return {
            "normed_cloud":   torch.from_numpy(normed_cloud),
            "raw_cloud":      torch.from_numpy(raw_cloud),
            "mask":           torch.from_numpy(mask),
            "n_vars":         n_vars,
            "var_names":      real_var_names,
            "eq_tokens":      torch.tensor(eq_tokens, dtype=torch.long),
            "target_val":     torch.from_numpy(raw_cloud[:, -1]),
            "actual_formula": str(row['Prefix_Formula'])
        }


def prefix_collate_fn(batch):
    normed_clouds = torch.stack([item['normed_cloud'] for item in batch])
    raw_clouds = torch.stack([item['raw_cloud'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    targets = torch.stack([item['target_val'] for item in batch])

    eq_tokens = [item['eq_tokens'] for item in batch]
    eq_padded = torch.nn.utils.rnn.pad_sequence(
        eq_tokens, batch_first=True, padding_value=0
    )

    actual_formulas = [item['actual_formula'] for item in batch]
    n_vars = torch.tensor([item['n_vars'] for item in batch], dtype=torch.long)
    var_names = [item['var_names'] for item in batch]

    return {
        "normed_cloud":    normed_clouds,
        "raw_cloud":       raw_clouds,
        "mask":            masks,
        "n_vars":          n_vars,
        "var_names":       var_names,
        "eq_tokens":       eq_padded,
        "target_val":      targets,
        "actual_formulas": actual_formulas,
    }


if __name__ == "__main__":
    BASE_DIR = "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr"
    CSV_PATH = os.path.join(BASE_DIR, "FeynmanFunctionalEquations.csv")
    DATA_PATH = os.path.join(BASE_DIR, "Feynman_with_units")

    build_vocab(CSV_PATH, "vocab.json")
    dataset = FeynmanDataset(CSV_PATH, DATA_PATH, "vocab.json", n_points=100)

    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"Cloud Shape: {batch['data_cloud'].shape}")
        print(f"Target Shape: {batch['raw_target'].shape}")

        sample_eq = batch['eq_tokens'][0]
        decoded = [dataset.id_to_word[idx.item()] for idx in sample_eq if idx != 0]
        print(f"Decoded: {''.join(decoded)}")