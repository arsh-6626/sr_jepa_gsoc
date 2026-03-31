import pandas as pd
import sympy
import json
import re
import os
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor, implicit_application
)

INPUT_FILE = "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/FeynmanEquations.csv"
OUTPUT_CSV = "/media/kavinder/hdd/ARSH_ARNABI/lmjepa_sr/FeynmanPrefixEquations.csv"
VOCAB_FILE = "../vocab_bfgs.json"

SYMBOLS_REQUIRING_SYM_SUFFIX = [
    r'Volt', r'mob', r'mom', r'Bx', r'By', r'Bz', r'Nn',
    r'Int_0', r'k_spring', r'mu_drift', r'rho_c_0', r'sigma_den', r'A_vec',
    r'omega_0', r'p_d', r'n_0', r'n_rho', r'm_rho', r'g_', r'kb', r'Ef', r'Pwr',
    r'm_0', r'q1', r'q2', r'I', r'I1', r'I2', r'pr', r'pi', r'e', r'gamma', r'beta'
]

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    implicit_application
)

def clean_formula_string(formula):
    for sym_name in SYMBOLS_REQUIRING_SYM_SUFFIX:
        formula = re.sub(r'\b' + sym_name + r'\b', sym_name + '_sym', formula)
    formula = re.sub(r'\b([VTrxyzmdtheta])([0-9])\b', r'\1\2_sym', formula)
    return formula

def to_prefix(expr, skeletal=True):
    if expr.is_Symbol:
        name = re.sub(r'_sym$', '', str(expr))
        return ["id", name]
    if expr.is_Number:
        val = "C" if skeletal else str(round(float(expr), 6))
        return ["const", val]
    if isinstance(expr, sympy.Function):
        op = expr.func.__name__.lower()
        res = [op]
        for arg in expr.args:
            res.extend(to_prefix(arg, skeletal))
        return res
    op_map = {'Add': 'add', 'Mul': 'mul', 'Pow': 'pow'}
    op_name = type(expr).__name__
    op = op_map.get(op_name, op_name.lower())
    
    args = list(expr.args)
    
    if len(args) > 2:
        left = to_prefix(args[0], skeletal)
        right = to_prefix(expr.func(*args[1:]), skeletal)
        return [op] + left + right
    
    res = [op]
    for arg in args:
        res.extend(to_prefix(arg, skeletal))
    return res

if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)
    prefix_formulas = []
    all_tokens = set()

    for idx, row in df.iterrows():
        raw = str(row['Formula']).replace("^", "**")
        cleaned = clean_formula_string(raw)
        
        potential_symbols = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cleaned)
        shield_ns = {s: sympy.Symbol(s) for s in potential_symbols}
        shield_ns['pi_sym'] = sympy.pi
        shield_ns['e_sym'] = sympy.E

        try:
            expr = parse_expr(cleaned, transformations=transformations, 
                            local_dict=shield_ns, evaluate=False)
            
            prefix_list = to_prefix(expr, skeletal=True)
            
            prefix_str = " ".join(prefix_list)
            prefix_formulas.append(prefix_str)
            
            all_tokens.update(prefix_list)
            
        except Exception as e:
            prefix_formulas.append(None)

    df['Prefix_Formula'] = prefix_formulas
    df['Formula'] = df['Prefix_Formula'] 
    
    df.to_csv(OUTPUT_CSV, index=False)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    for i, token in enumerate(sorted(list(all_tokens))):
        vocab[token] = i + 4

    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f, indent=4)
    
    print(f"Vocab size: {len(vocab)}")