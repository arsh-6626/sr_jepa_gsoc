import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import sys
import os
import re
import copy
import random
import numpy as np
import sympy
from sympy import sympify, symbols, lambdify
from scipy.optimize import minimize
sys.path.append("/media/kavinder/hdd/ARSH_ARNABI/jepa_sr/")
from src.models.decoder_sparse_jepa import SR_JEPA_Sparse_Decoder
from src.datasets.feynman_dataset import FeynmanPrefixDataset, prefix_collate_fn


CONFIG = {
    "learning_rate": 1e-4,
    "max_lr": 4e-4,
    "epochs": 200,
    "batch_size": 64,
    "dim_size": 256,
    "n_layers": 2,
    "n_head": 4,
    "lambda_jepa_max": 1.1,
    "jepa_warmup_epochs": 5,
    "jepa_dropout": 0.3,
    "k_tokens": 2,
    "val_split": 0.2,
    "checkpoint_dir": "checkpoints_prefix__sparse_final",
    "window": 16,
    "bfgs_restarts": 16,
}

OPERATOR_TOKENS = {
    "add", "mul", "sub", "div", "pow",
    "sin", "cos", "exp", "log", "sqrt", "neg", "abs",
    "tanh", "asin", "acos", "atan",
}

NON_VARIABLE_TOKENS = OPERATOR_TOKENS | {
    "const", "pi", "e",
    "[BOS]", "[EOS]", "[PAD]", "[PRED]", "[UNK]",
    "id",
}


class SymbolicEvaluator:
    def __init__(self, id_to_word):
        self.id_to_word = id_to_word
        self.sym_names = [
            "x", "y", "z", "v", "t", "r", "p", "m", "theta", "sigma", "theta1",
            "x1", "x2", "y1", "y2", "z1", "z2", "m1", "m2", "G", "m_0", "c",
            "mu", "Nn", "q1", "q2", "epsilon", "omega", "Ef", "B", "mom", "omega_0",
            "q", "n", "k", "h", "kb", "alpha", "rho", "Bx", "By", "Bz",
            "kappa", "gamma", "delta", "lambda", "nu", "tau", "phi",
            "rho_c_0", "rho_c", "v1", "v2", "u", "w", "F", "L", "I", "d",
            "n1", "n2", "q_0", "omega1", "omega2", "r1", "r2",
        ]
        self.sym_map = {name: symbols(name) for name in self.sym_names}
        self.sym_map['pi'] = sympy.pi
        self.sym_map['e']  = sympy.E

    def _recursive_prefix_to_sympy(self, tokens):
        if not tokens:
            return None
        token = tokens.pop(0)

        if token.startswith("VAR_"):
            var_name = token[4:]
            return self.sym_map.get(var_name, symbols(var_name))

        if token.startswith("CONST_"):
            idx = int(token[6:])
            return symbols(f"C{idx}")

        if token.startswith("NUM_"):
            try:
                return sympify(float(token[4:]))
            except Exception:
                return symbols(token[4:])

        if token in ["add", "mul", "sub", "div", "pow"]:
            left  = self._recursive_prefix_to_sympy(tokens)
            right = self._recursive_prefix_to_sympy(tokens)
            if left is None or right is None:
                return None
            if token == "add": return left + right
            if token == "mul": return left * right
            if token == "sub": return left - right
            if token == "div": return left / right if right != 0 else None
            if token == "pow": return left ** right

        if token in ["sin", "cos", "exp", "log", "sqrt", "neg", "abs", "tanh", "asin", "acos", "atan"]:
            arg = self._recursive_prefix_to_sympy(tokens)
            if arg is None:
                return None
            if token == "sin":  return sympy.sin(arg)
            if token == "cos":  return sympy.cos(arg)
            if token == "exp":  return sympy.exp(arg)
            if token == "log":  return sympy.log(arg)
            if token == "sqrt": return sympy.sqrt(arg)
            if token == "neg":  return -arg
            if token == "abs":  return sympy.Abs(arg)
            if token == "tanh": return sympy.tanh(arg)
            if token == "asin": return sympy.asin(arg)
            if token == "acos": return sympy.acos(arg)
            if token == "atan": return sympy.atan(arg)

        if token == "pi":  return sympy.pi
        if token == "e":   return sympy.E

        return None

    def clean_to_sympy(self, token_ids, pad_id, eos_id):
        words = []
        for i in token_ids:
            idx = i.item()
            if idx == eos_id:
                break
            if idx == pad_id:
                continue
            word = self.id_to_word.get(idx, "")
            if word in {"[BOS]", "[EOS]", "[PAD]", "[PRED]", ""}:
                continue
            if word.startswith("<|predictor_"):
                continue
            words.append(word)

        raw_tokens = []
        c_idx = 0
        for word in words:
            if word == "const":
                raw_tokens.append(f"CONST_{c_idx}")
                c_idx += 1
            elif word in OPERATOR_TOKENS or word in {"pi", "e"}:
                raw_tokens.append(word)
            elif re.match(r'^-?[\d\.]+([eE][+-]?\d+)?$', word):
                raw_tokens.append(f"NUM_{word}")
            elif word in {"[BOS]", "[EOS]", "[PAD]", "[PRED]", "[UNK]"}:
                continue
            else:
                if word not in OPERATOR_TOKENS:
                    raw_tokens.append(f"VAR_{word}")

        try:
            expr = self._recursive_prefix_to_sympy(raw_tokens[:])
            return expr, c_idx
        except Exception:
            return None, 0

    def _bfgs_optimize(self, pred_expr, const_syms, var_syms, x_inputs,
                       y_true, n_real, n_restarts):
        f_numeric = lambdify(const_syms + var_syms, pred_expr, "numpy")
        y_norm_sq = float(np.dot(y_true, y_true)) + 1e-8

        def objective(params):
            try:
                y_pred = f_numeric(*params, *x_inputs)
                y_pred = np.asarray(y_pred, dtype=np.float64)
                if y_pred.shape == ():
                    y_pred = np.full(n_real, float(y_pred))
                if y_pred.shape != (n_real,) or not np.isfinite(y_pred).all():
                    return 1e10
                return float(np.mean((y_true - y_pred) ** 2)) / y_norm_sq
            except Exception:
                return 1e10

        use_jac = None
        try:
            grad_exprs = [sympy.diff(pred_expr, s) for s in const_syms]
            grad_funcs = [
                lambdify(const_syms + var_syms, g, "numpy") for g in grad_exprs
            ]

            def jacobian(params):
                try:
                    y_pred = f_numeric(*params, *x_inputs)
                    y_pred = np.asarray(y_pred, dtype=np.float64)
                    if y_pred.shape == ():
                        y_pred = np.full(n_real, float(y_pred))
                    residual = y_true - y_pred
                    grads = []
                    for gf in grad_funcs:
                        gv = np.asarray(gf(*params, *x_inputs), dtype=np.float64)
                        if gv.shape == ():
                            gv = np.full(n_real, float(gv))
                        if gv.shape != (n_real,) or not np.isfinite(gv).all():
                            return None
                        grads.append(
                            float(np.mean(-2.0 * residual * gv)) / y_norm_sq
                        )
                    return np.array(grads)
                except Exception:
                    return None

            use_jac = jacobian
        except Exception:
            pass

        rng    = np.random.default_rng(seed=42)
        starts = [np.ones(len(const_syms))]
        for _ in range(n_restarts - 1):
            scale = rng.uniform(-2, 2, size=len(const_syms))
            signs = rng.choice([-1.0, 1.0], size=len(const_syms))
            starts.append(signs * (10.0 ** scale))

        best_norm_mse = 1e10
        for x0 in starts:
            try:
                res = minimize(
                    objective, x0=x0, method='BFGS', jac=use_jac,
                    tol=1e-8, options={"maxiter": 1000},
                )
                if res.fun < best_norm_mse:
                    best_norm_mse = float(res.fun)
            except Exception:
                continue

        return best_norm_mse * y_norm_sq

    def calculate_metrics(
        self,
        pred_ids,
        raw_data_cloud,
        target_vals,
        mask,
        var_names,
        pad_id,
        eos_id,
        n_restarts=8,
    ):
        hit, r2, final_mse = 0, 0.0, 1.0
        bfgs_succeeded = False

        n_real = int(mask.sum().item())
        if n_real == 0:
            return hit, r2, final_mse, bfgs_succeeded

        real_cloud = raw_data_cloud[:n_real].cpu().numpy()
        y_true     = target_vals[:n_real].cpu().numpy()

        if np.var(y_true) < 1e-12:
            return hit, r2, final_mse, bfgs_succeeded

        pred_expr, num_consts = self.clean_to_sympy(pred_ids, pad_id, eos_id)
        if pred_expr is None:
            return hit, r2, final_mse, bfgs_succeeded

        try:
            free_syms  = list(pred_expr.free_symbols)
            const_syms = sorted(
                [s for s in free_syms if str(s).startswith('C')],
                key=lambda s: int(str(s)[1:])
            )
            var_syms = [s for s in free_syms if not str(s).startswith('C')]

            col_indices = []
            missing_vars = []
            for s in var_syms:
                name = str(s)
                if name in var_names:
                    col_indices.append(var_names.index(name))
                else:
                    missing_vars.append(name)

            if missing_vars:
                return hit, r2, final_mse, bfgs_succeeded

            x_inputs = [real_cloud[:, i] for i in col_indices]

            if num_consts > 0:
                final_mse = self._bfgs_optimize(
                    pred_expr, const_syms, var_syms, x_inputs,
                    y_true, n_real, n_restarts,
                )
            else:
                f_numeric = lambdify(var_syms, pred_expr, "numpy")
                y_pred    = np.asarray(f_numeric(*x_inputs), dtype=np.float64)
                if y_pred.shape == ():
                    y_pred = np.full(n_real, float(y_pred))
                if y_pred.shape != (n_real,) or not np.isfinite(y_pred).all():
                    return hit, r2, final_mse, bfgs_succeeded
                final_mse = float(np.mean((y_true - y_pred) ** 2))

            bfgs_succeeded = True
            var_y = float(np.var(y_true)) + 1e-8
            r2    = float(max(0.0, 1.0 - (final_mse / var_y)))
            if final_mse < 1e-7:
                hit = 1

        except Exception:
            pass

        return hit, r2, final_mse, bfgs_succeeded


def apply_variable_constraint(logits, var_names, word_to_id, device):
    B, V = logits.shape

    all_var_ids = []
    for word, idx in word_to_id.items():
        if word in NON_VARIABLE_TOKENS:
            continue
        if word.startswith("["):
            continue
        if re.match(r'^-?[\d\.]+([eE][+-]?\d+)?$', word):
            continue
        all_var_ids.append(idx)

    for b in range(B):
        allowed_ids = set()
        for name in var_names[b]:
            tok_id = word_to_id.get(name)
            if tok_id is not None:
                allowed_ids.add(tok_id)

        forbidden_ids = [idx for idx in all_var_ids if idx not in allowed_ids]
        if forbidden_ids:
            logits[b, forbidden_ids] = -1e9

    return logits


def levenshtein_distance(seq1: list, seq2: list) -> int:
    """Standard dynamic-programming Levenshtein distance between two token lists."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def validate(model, dataloader, device, config, vocab_size, pad_token_id, ds):
    model.eval()
    total_hits, total_r2, total_out_mse, total_samples = 0, 0.0, 0.0, 0
    total_exact_matches = 0
    total_tokens_correct = 0
    total_tokens_count = 0
    total_complexity = 0
    total_lev_acc = 0.0

    evaluator = SymbolicEvaluator(ds.id_to_word)
    eos_id = ds.word_to_id.get("[EOS]", -1)
    bos_id = ds.word_to_id.get("[BOS]", -1)

    table_data = []

    with torch.no_grad():
        for batch in dataloader:
            norm_cloud     = batch['normed_cloud'].to(device)
            raw_cloud      = batch['raw_cloud'].to(device)
            target_val     = batch['target_val'].to(device)
            actual_formulas = batch['actual_formulas']
            gt_token_ids   = batch['eq_tokens'].to(device)
            mask           = batch['mask'].to(device)
            var_names      = batch['var_names']

            batch_size  = norm_cloud.size(0)
            curr_tokens = torch.full((batch_size, 1), bos_id, device=device)
            generated_ids = torch.full((batch_size, 50), pad_token_id, dtype=torch.long, device=device)
            finished    = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(50):
                logits      = model.decode_step(norm_cloud, curr_tokens, var_names, pad_mask=mask)
                last_logits = logits[:, -1, :].clone()
                last_logits = apply_variable_constraint(last_logits, var_names, ds.word_to_id, device)

                next_token = torch.argmax(last_logits, dim=-1)
                next_token = torch.where(finished, torch.tensor(pad_token_id, device=device), next_token)

                generated_ids[:, step] = next_token
                curr_tokens = torch.cat([curr_tokens, next_token.unsqueeze(1)], dim=1)
                finished |= (next_token == eos_id)
                if finished.all():
                    break

            for i in range(batch_size):
                # ── Build pred_trimmed ───────────────────────────────────────
                pred_seq     = generated_ids[i].tolist()
                pred_trimmed = []
                for t in pred_seq:
                    if t == eos_id:
                        break
                    if t != pad_token_id:
                        pred_trimmed.append(t)

                # ── Build gt_trimmed ─────────────────────────────────────────
                gt_seq     = gt_token_ids[i].tolist()
                gt_trimmed = [t for t in gt_seq if t not in [bos_id, eos_id, pad_token_id]]

                # ── Levenshtein (now safe — both lists are defined) ───────────
                max_len   = max(len(pred_trimmed), len(gt_trimmed), 1)
                lev_dist  = levenshtein_distance(pred_trimmed, gt_trimmed)
                total_lev_acc += 1.0 - lev_dist / max_len

                # ── Exact match ──────────────────────────────────────────────
                if pred_trimmed == gt_trimmed:
                    total_exact_matches += 1

                # ── Token-level accuracy ─────────────────────────────────────
                min_len = min(len(pred_trimmed), len(gt_trimmed))
                if min_len > 0:
                    correct = sum(
                        1 for p, g in zip(pred_trimmed[:min_len], gt_trimmed[:min_len]) if p == g
                    )
                    total_tokens_correct += correct
                    total_tokens_count   += max(len(pred_trimmed), len(gt_trimmed))

                total_complexity += len(pred_trimmed)

                # ── Numerical / symbolic eval ────────────────────────────────
                hit, r2, o_mse, _ = evaluator.calculate_metrics(
                    generated_ids[i], raw_cloud[i], target_val[i],
                    mask[i], var_names[i], pad_token_id, eos_id,
                    n_restarts=config["bfgs_restarts"]
                )
                total_hits    += hit
                total_r2      += r2
                total_out_mse += o_mse
                total_samples += 1

                pred_str = " ".join([ds.id_to_word.get(t, str(t)) for t in pred_trimmed])
                gt_str   = actual_formulas[i]
                table_data.append([gt_str, pred_str, f"{r2:.4f}", "✅" if hit == 1 else "❌"])

    metrics = {
        "val/r2_score":                       total_r2      / max(total_samples, 1),
        "val/mse":                            total_out_mse / max(total_samples, 1),
        "val/functional_equation_score":      (total_hits / max(total_samples, 1)) * 100,
        "val/exact_match_pct":                (total_exact_matches / max(total_samples, 1)) * 100,
        "val/token_accuracy_pct":             (total_tokens_correct / max(total_tokens_count, 1)) * 100,
        "val/structural_levenshtein_accuracy": total_lev_acc / max(total_samples, 1),
    }

    sample_table = wandb.Table(columns=["Target Formula", "Predicted Prefix", "R2", "Hit"])
    for row in table_data[:32]:
        sample_table.add_data(*row)
    metrics["val/predictions_debug"] = sample_table

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def train():
    config = CONFIG
    wandb.init(project="SR-JEPA-Feynman", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    ds = FeynmanPrefixDataset(
        "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/FeynmanAugmentPrefixEquations.csv",
        "/home/kavinder/Feynman_Augmented",
        "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/sr_jepa/vocab_bfgs_prefix.json",
    )

    predictor_tokens = ["[PRED]"] * config["k_tokens"]
    for token in predictor_tokens:
        if token not in ds.word_to_id:
            new_id = len(ds.word_to_id)
            ds.word_to_id[token] = new_id
            ds.id_to_word[new_id] = token

    val_sz   = int(len(ds) * config["val_split"])
    train_ds, val_ds = random_split(ds, [len(ds) - val_sz, val_sz])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=prefix_collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        collate_fn=prefix_collate_fn, num_workers=0, pin_memory=True,
    )

    vocab_size = len(ds.word_to_id)
    pad_id     = ds.word_to_id["[PAD]"]

    model = SR_JEPA_Sparse_Decoder(
        vocab_size,
        config["dim_size"],
        config["n_head"],
        config["n_layers"],
        word_to_id=ds.word_to_id,
        d_in=13,
        k_tokens=config["k_tokens"],
        window=config["window"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
    )

    best_val_score = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        pbar        = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        curr_lambda = config["lambda_jepa_max"] * min(
            1.0, epoch / max(config["jepa_warmup_epochs"], 1)
        )

        for batch in pbar:
            norm_cloud = batch['normed_cloud'].to(device)
            eq_tokens  = batch['eq_tokens'].to(device)
            mask       = batch['mask'].to(device)
            var_names  = batch["var_names"]
            run_jepa   = random.random() > config["jepa_dropout"]

            if run_jepa:
                logits, sy_tilde, sy, data_end_idx = model(norm_cloud, eq_tokens, var_names, pad_mask=mask)
            else:
                logits, _, _, data_end_idx = model(norm_cloud, eq_tokens, var_names, pad_mask=mask)

            eq_logits  = logits[:, data_end_idx:, :]
            lm_targets = eq_tokens[:, 1:]

            min_len    = min(eq_logits.size(1), lm_targets.size(1))
            eq_logits  = eq_logits[:, :min_len, :]
            lm_targets = lm_targets[:, :min_len]

            loss_lm = F.cross_entropy(
                eq_logits.reshape(-1, vocab_size),
                lm_targets.reshape(-1),
                ignore_index=pad_id,
                label_smoothing=0.1,
            )

            if run_jepa:
                loss_jepa = 1 - F.cosine_similarity(sy_tilde, sy.detach()).mean()
                loss = loss_lm + curr_lambda * loss_jepa
            else:
                loss_jepa = torch.tensor(0.0)
                loss = loss_lm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train/loss":      loss.item(),
                "train/lm_loss":   loss_lm.item(),
                "train/jepa_loss": loss_jepa.item(),
            })
            pbar.set_postfix({"LM": f"{loss_lm.item():.2f}"})

        val_res = validate(model, val_loader, device, config, vocab_size, pad_id, ds)

        wandb.log({"epoch": epoch + 1, **val_res})

        val_score = val_res.get("val/functional_equation_score", 0.0)
        if val_score > best_val_score:
            best_val_score = val_score
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(config["checkpoint_dir"], "best_model.pt")
            )

    wandb.finish()


if __name__ == "__main__":
    train()
