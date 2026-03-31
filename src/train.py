import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import sys
import os
import random
import numpy as np
import sympy
from sympy import sympify, simplify, symbols, lambdify
from scipy.optimize import minimize
sys.path.append("/media/kavinder/hdd/ARSH_ARNABI/jepa_sr")
from src.models.decoder_only_jepa import SR_JEPA_Decoder
from src.embeddings.tnet_embeds import tnet_reg_loss
from src.datasets.feynman_dataset import FeynmanPrefixDataset, prefix_collate_fn

CONFIG = {
    "learning_rate": 1e-4,
    "max_lr": 4e-4,
    "epochs": 200,
    "batch_size": 8,
    "dim_size": 512,
    "n_layers": 6,
    "n_head": 8,
    "lambda_jepa_max": 1.0,
    "jepa_warmup_epochs": 15,
    "jepa_dropout": 0.2,
    "k_tokens": 6,
    "val_split": 0.2,
    "checkpoint_dir": "checkpoints_prefix_final",
    "tnet_reg_weight": 1e-3
}

def get_n_points(n_vars):
    if n_vars == 1:
        return 100
    elif n_vars == 2:
        return 200
    elif n_vars == 3:
        return 500
    else:
        return min(1000, 100 * (2 ** (n_vars - 1)))
    

class SymbolicEvaluator:
    def __init__(self, id_to_word):
        self.id_to_word = id_to_word
        # sym_names still needed for sympy symbol creation,
        # but NO LONGER used for column index mapping
        self.sym_names = [
            "x", "y", "z", "v", "t", "r", "p", "m", "theta", "sigma", "theta1",
            "x1", "x2", "y1", "y2", "z1", "z2", "m1", "m2", "G", "m_0", "c",
            "mu", "Nn", "q1", "q2", "epsilon", "omega", "Ef", "B", "mom", "omega_0"
        ]
        self.sym_map = {name: symbols(name) for name in self.sym_names}
        self.sym_map['pi'] = sympy.pi
        self.sym_map['e']  = sympy.E
 
    def _recursive_prefix_to_sympy(self, tokens):
        if not tokens:
            return None
        token = tokens.pop(0)
        if token == "id":
            var_name = tokens.pop(0) if tokens else "x"
            return self.sym_map.get(var_name, symbols(var_name))
        if token == "const":
            val = tokens.pop(0) if tokens else "1.0"
            if val.startswith("C"):
                return symbols(val)
            try:
                return sympify(float(val))
            except ValueError:
                return symbols(val)
        if token in ["add", "mul", "sub", "div", "pow"]:
            left  = self._recursive_prefix_to_sympy(tokens)
            right = self._recursive_prefix_to_sympy(tokens)
            if left is None or right is None:
                return None
            if token == "add": return left + right
            if token == "mul": return left * right
            if token == "sub": return left - right
            if token == "div": return left / right
            if token == "pow": return left ** right
        if token in ["sin", "cos", "exp", "log", "sqrt", "neg", "abs"]:
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
        return None
 
    def clean_to_sympy(self, token_ids, pad_id, eos_id):
        raw_tokens = []
        c_idx = 0
        for i in token_ids:
            idx = i.item()
            if idx == pad_id:
                continue
            if idx == eos_id:
                break
            word = self.id_to_word.get(idx, "")
            if (word in ["[BOS]", "[EOS]", "[PAD]", "[PRED]", ""]
                    or word.startswith("<|predictor_")):
                continue
            if word in ["C", "[CONST]", "const([CONST])"]:
                raw_tokens.extend(["const", f"C{c_idx}"])
                c_idx += 1
            else:
                raw_tokens.append(word)
        try:
            expr = self._recursive_prefix_to_sympy(raw_tokens[:])
            return expr, c_idx
        except Exception:
            return None, 0
 
    def calculate_metrics(
        self,
        pred_ids,
        raw_data_cloud,   # (max_points, max_vars+1) tensor — may contain pad zeros
        target_vals,      # (max_points,)             tensor — may contain pad zeros
        mask,             # (max_points,)             float tensor, 1=real 0=pad
        var_names,        # list of str, variable names in file column order
                          # e.g. ["x", "y"] → col0=x, col1=y, last col=target
        pad_id,
        eos_id,
    ):
        """
        Returns (hit, r2, final_mse).
 
        hit      : 1 if final_mse < 1e-7 (expression numerically matches data)
        r2       : coefficient of determination on real (non-padded) points
        final_mse: MSE on real points after BFGS constant optimisation
        """
        hit, r2, final_mse = 0, 0.0, 1.0
 
        # --- 1. Slice real points only using mask ---
        n_real = int(mask.sum().item())
        if n_real == 0:
            return hit, r2, final_mse
 
        real_cloud = raw_data_cloud[:n_real].cpu().numpy()   # (n_real, max_vars+1)
        y_true     = target_vals[:n_real].cpu().numpy()       # (n_real,)
 
        if np.var(y_true) < 1e-12:
            # target is essentially constant — R2 is undefined, skip
            return hit, r2, final_mse
 
        # --- 2. Parse predicted expression ---
        pred_expr, num_consts = self.clean_to_sympy(pred_ids, pad_id, eos_id)
        if pred_expr is None:
            return hit, r2, final_mse
 
        try:
            free_syms  = list(pred_expr.free_symbols)
            const_syms = sorted(
                [s for s in free_syms if str(s).startswith('C')],
                key=lambda s: int(str(s)[1:])
            )
            var_syms = [s for s in free_syms if not str(s).startswith('C')]
 
            # --- 3. Map var_syms to columns using actual file column order ---
            # var_names[i] = name of variable in column i of real_cloud
            # last column is the target, so input columns are 0..len(var_names)-1
            col_indices = []
            for s in var_syms:
                name = str(s)
                if name in var_names:
                    col_indices.append(var_names.index(name))
                else:
                    # variable name in expression not found in this equation's
                    # variable list — expression is wrong, bail out
                    return hit, r2, final_mse
 
            x_inputs = [real_cloud[:, i] for i in col_indices]  # list of (n_real,)
 
            # --- 4. Evaluate or optimise constants ---
            if num_consts > 0:
                f_numeric = lambdify(const_syms + var_syms, pred_expr, "numpy")
 
                def objective(params):
                    try:
                        y_pred = f_numeric(*params, *x_inputs)
                        # validate shape and finiteness
                        y_pred = np.asarray(y_pred, dtype=np.float64)
                        if y_pred.shape == ():
                            y_pred = np.full(n_real, float(y_pred))
                        if y_pred.shape != (n_real,):
                            return 1e10
                        if not np.isfinite(y_pred).all():
                            return 1e10
                        return float(np.mean((y_true - y_pred) ** 2))
                    except Exception:
                        return 1e10
 
                res = minimize(
                    objective,
                    x0=[1.0] * len(const_syms),
                    method='BFGS',
                    tol=1e-6,
                )
                final_mse = float(res.fun)
 
            else:
                f_numeric = lambdify(var_syms, pred_expr, "numpy")
                y_pred = f_numeric(*x_inputs)
                y_pred = np.asarray(y_pred, dtype=np.float64)
 
                # handle scalar broadcast (e.g. expression simplified to a constant)
                if y_pred.shape == ():
                    y_pred = np.full(n_real, float(y_pred))
 
                if y_pred.shape != (n_real,):
                    return hit, r2, final_mse
                if not np.isfinite(y_pred).all():
                    return hit, r2, final_mse
 
                final_mse = float(np.mean((y_true - y_pred) ** 2))
 
            # --- 5. Compute R2 ---
            var_y = float(np.var(y_true)) + 1e-8
            r2    = float(max(0.0, 1.0 - (final_mse / var_y)))
            if final_mse < 1e-7:
                hit = 1
 
        except Exception:
            pass
 
        return hit, r2, final_mse


def validate(model, dataloader, device, config, vocab_size, pad_token_id, ds):
    model.eval()
    total_hits, total_r2, total_out_mse, total_samples = 0, 0.0, 0.0, 0

    evaluator = SymbolicEvaluator(ds.id_to_word)
    eos_id = ds.word_to_id.get("[EOS]", -1)
    bos_id = ds.word_to_id.get("[BOS]", -1)

    html_rows = []
    max_examples = 5

    with torch.no_grad():
        for batch in dataloader:
            norm_cloud      = batch['normed_cloud'].to(device)
            raw_cloud       = batch['raw_cloud'].to(device)
            target_val      = batch['target_val'].to(device)
            actual_formulas = batch['actual_formulas']
            mask            = batch['mask'].to(device)  # (B, max_points)

            batch_size = norm_cloud.size(0)
            curr_tokens = torch.full((batch_size, 1), bos_id, device=device)
            generated_ids = torch.full((batch_size, 50), pad_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(50):
                logits = model.decode_step(norm_cloud, curr_tokens, mask=mask)

                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                next_token = torch.where(
                    finished,
                    torch.tensor(pad_token_id, device=device),
                    next_token
                )
                generated_ids[:, step] = next_token
                curr_tokens = torch.cat([curr_tokens, next_token.unsqueeze(1)], dim=1)
                finished |= (next_token == eos_id)
                if finished.all():
                    break

            for i in range(batch_size):
                hit, r2, o_mse = evaluator.calculate_metrics(
                    generated_ids[i],
                    raw_cloud[i],
                    target_val[i],
                    mask[i],                 # ✅ add this
                    batch["var_names"][i],   # ✅ add this (see note below)
                    pad_token_id,
                    eos_id
                )
                total_hits     += hit
                total_r2       += r2
                total_out_mse  += o_mse
                total_samples  += 1

                if len(html_rows) < max_examples:
                    pred_tokens = [
                        ds.id_to_word.get(int(t.item()), "")
                        for t in generated_ids[i]
                        if int(t.item()) != pad_token_id
                    ]
                    html_rows.append(f"""
                    <tr>
                        <td>{actual_formulas[i]}</td>
                        <td>{" ".join(pred_tokens)}</td>
                        <td>{r2:.3f}</td>
                    </tr>
                    """)

    metrics = {
        "output_mse": total_out_mse / total_samples,
        "functional_equation_score": (total_hits / total_samples) * 100,
        "mean_r2": total_r2 / total_samples,
    }
    html_output = f"""
    <table border="1" style="border-collapse: collapse;">
        <tr><th>Ground Truth</th><th>Prediction</th><th>R2</th></tr>
        {''.join(html_rows)}
    </table>
    """
    return metrics, html_output


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def train():
    config = CONFIG
    wandb.init(project="SR-JEPA-Feynman", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    ds = FeynmanPrefixDataset(
        "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/FeynmanPrefixEquations.csv",
        "/home/kavinder/Feynman_with_units",
        "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/sr_jepa/vocab_bfgs_prefix.json"
    )

    predictor_tokens = [f"[PRED]" for i in range(config["k_tokens"])]
    for token in predictor_tokens:
        if token not in ds.word_to_id:
            new_id = len(ds.word_to_id)
            ds.word_to_id[token] = new_id
            ds.id_to_word[new_id] = token

    val_sz   = int(len(ds) * config["val_split"])
    train_ds, val_ds = random_split(ds, [len(ds) - val_sz, val_sz])
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=prefix_collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        collate_fn=prefix_collate_fn, num_workers=0, pin_memory=True
    )

    vocab_size = len(ds.word_to_id)
    pad_id     = ds.word_to_id["[PAD]"]

    model = SR_JEPA_Decoder(
        vocab_size,
        config["dim_size"],
        config["n_head"],
        config["n_layers"],
        d_in=13,
        k_tokens=config["k_tokens"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"]
    )

    best_val_score = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        curr_lambda = config["lambda_jepa_max"] * min(
            1.0, epoch / config["jepa_warmup_epochs"]
        )

        for batch in pbar:
            norm_cloud = batch['normed_cloud'].to(device)
            eq_tokens  = batch['eq_tokens'].to(device)
            mask       = batch['mask'].to(device)  # (B, max_points)

            run_jepa = random.random() > config["jepa_dropout"]

            if run_jepa:
                logits, sy_tilde, sy = model(norm_cloud, eq_tokens, mask=mask)
            else:
                logits, _, _ = model(norm_cloud, eq_tokens, mask=mask)

            data_end_idx = 1 + config["k_tokens"]
            eq_logits  = logits[:, data_end_idx:-1, :]
            lm_targets = eq_tokens[:, 1:]

            loss_lm = F.cross_entropy(
                eq_logits.reshape(-1, vocab_size),
                lm_targets.reshape(-1),
                ignore_index=pad_id
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

        val_res, val_html = validate(
            model, val_loader, device, config, vocab_size, pad_id, ds
        )

        if val_res["functional_equation_score"] > best_val_score:
            best_val_score = val_res["functional_equation_score"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_res,
                os.path.join(config["checkpoint_dir"], "best_model.pt")
            )

        wandb.log({
            "epoch": epoch + 1,
            **{f"val/{k}": v for k, v in val_res.items()},
            "val/examples": wandb.Html(val_html),
        })

    wandb.finish()


if __name__ == "__main__":
    train()