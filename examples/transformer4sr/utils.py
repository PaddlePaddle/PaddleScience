# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""

from typing import List

import numpy as np
import paddle
import sympy
import yaml
import zss
from typing_extensions import Literal

with open("./conf/transformer4sr.yaml", "r") as file:
    cfg = yaml.safe_load(file)
vocab_library = cfg["DATA"]["vocab_library"]

C, x1, x2, x3, x4, x5, x6 = sympy.symbols(
    "C, x1, x2, x3, x4, x5, x6", real=True, positive=True
)
MY_VOCAB = np.array(
    [
        ["add", 4, 2],  # binary operators
        ["sub", 3, 2],
        ["mul", 6, 2],
        ["sin", 1, 1],  # unary operators
        ["cos", 1, 1],
        ["log", 2, 1],
        ["exp", 2, 1],
        ["neg", 0, 1],
        ["inv", 3, 1],
        ["sq", 2, 1],
        ["cb", 0, 1],
        ["sqrt", 2, 1],
        ["cbrt", 0, 1],
        ["C", 8, 0],  # leaves
        ["x1", 8, 0],
        ["x2", 8, 0],
        ["x3", 4, 0],
        ["x4", 4, 0],
        ["x5", 2, 0],
        ["x6", 2, 0],
    ]
)


def from_seq_to_sympy(expr):
    """
    Recursive function!
    Convert the initial sequence of tokens into SymPy expression.
    """
    cur_token = expr[0]
    try:
        return float(cur_token)
    except ValueError:
        cur_idx = np.where(MY_VOCAB[:, 0] == cur_token)[0][0]
        cur_arity = int(MY_VOCAB[cur_idx, 2])
    if cur_arity == 0:
        if cur_token == "C":
            return C
        elif cur_token == "x1":
            return x1
        elif cur_token == "x2":
            return x2
        elif cur_token == "x3":
            return x3
        elif cur_token == "x4":
            return x4
        elif cur_token == "x5":
            return x5
        elif cur_token == "x6":
            return x6
    elif cur_arity == 1:
        if cur_token == "sin":
            return sympy.sin(from_seq_to_sympy(expr[1:]))
        elif cur_token == "cos":
            return sympy.cos(from_seq_to_sympy(expr[1:]))
        elif cur_token == "log":
            return sympy.log(from_seq_to_sympy(expr[1:]))
        elif cur_token == "exp":
            return sympy.exp(from_seq_to_sympy(expr[1:]))
        elif cur_token == "neg":
            return -from_seq_to_sympy(expr[1:])
        elif cur_token == "inv":
            return 1 / from_seq_to_sympy(expr[1:])
        elif cur_token == "sq":
            return from_seq_to_sympy(expr[1:]) ** 2
        elif cur_token == "cb":
            return from_seq_to_sympy(expr[1:]) ** 3
        elif cur_token == "sqrt":
            return sympy.sqrt(from_seq_to_sympy(expr[1:]))
        elif cur_token == "cbrt":
            return sympy.cbrt(from_seq_to_sympy(expr[1:]))
    elif cur_arity == 2:
        arity_count = 1
        idx_split = 1
        for temp_token in expr[1:]:
            try:
                float(temp_token)
                arity_count += -1
            except ValueError:
                temp_idx = np.where(MY_VOCAB[:, 0] == temp_token)[0][0]
                arity_count += int(MY_VOCAB[temp_idx, 2]) - 1
            idx_split += 1
            if arity_count == 0:
                break
        left_list = expr[1:idx_split]
        right_list = expr[idx_split:]
        if cur_token == "add":
            return from_seq_to_sympy(left_list) + from_seq_to_sympy(right_list)
        elif cur_token == "sub":
            return from_seq_to_sympy(left_list) - from_seq_to_sympy(right_list)
        elif cur_token == "mul":
            return from_seq_to_sympy(left_list) * from_seq_to_sympy(right_list)


def from_sympy_to_seq(sympy_expr):
    """
    Recursive function!
    Convert a SymPy expression into a standardized sequence of tokens.
    This function calls from_sympy_power_to_seq,
    from_sympy_mul_to_seq, and from_sympy_addition_to sequence.
    """
    if len(sympy_expr.args) == 0:  # leaf
        return [str(sympy_expr)]
    elif len(sympy_expr.args) == 1:  # unary operator
        return [str(sympy_expr.func)] + from_sympy_to_seq(sympy_expr.args[0])
    elif len(sympy_expr.args) >= 2:  # binary operator
        if sympy_expr.func == sympy.core.power.Pow:
            power_seq = from_sympy_power_to_seq(sympy_expr.args[1])
            return power_seq + from_sympy_to_seq(sympy_expr.args[0])
        elif sympy_expr.func == sympy.core.mul.Mul:
            return from_sympy_mul_to_seq(sympy_expr)
        elif sympy_expr.func == sympy.core.add.Add:
            return from_sympy_add_to_seq(sympy_expr)


def from_sympy_power_to_seq(exponent):
    """
    C.f. from_sympy_to_seq function.
    Standardize the sequence of tokens for power functions.
    """
    if exponent == (-4):
        return ["inv", "sq", "sq"]
    elif exponent == (-3):
        return ["inv", "cb"]
    elif exponent == (-2):
        return ["inv", "sq"]
    elif exponent == (-3 / 2):
        return ["inv", "cb", "sqrt"]
    elif exponent == (-1):
        return ["inv"]
    elif exponent == (-1 / 2):
        return ["inv", "sqrt"]
    elif exponent == (-1 / 3):
        return ["inv", "cbrt"]
    elif exponent == (-1 / 4):
        return ["inv", "sqrt", "sqrt"]
    elif exponent == (1 / 4):
        return ["sqrt", "sqrt"]
    elif exponent == (1 / 3):
        return ["cbrt"]
    elif exponent == (1 / 2):
        return ["sqrt"]
    elif exponent == (3 / 2):
        return ["cb", "sqrt"]
    elif exponent == (2):
        return ["sq"]
    elif exponent == (3):
        return ["cb"]
    elif exponent == (4):
        return ["sq", "sq"]
    else:
        return ["abort"]


def from_sympy_mul_to_seq(sympy_mul_expr):
    """
    C.f. from_sympy_to_seq function.
    Standardize the sequence of tokens for multiplications.
    """
    tokens = ["x1", "x2", "x3", "x4", "x5", "x6"]
    num_factors = 0
    num_constants = 0
    is_neg = False
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if cur_fact == (-1):
            is_neg = True
        if any(t in str(cur_fact) for t in tokens):
            num_factors += 1
        else:
            num_constants += 1
    seq = []
    if is_neg:
        seq.append("neg")
    for _ in range(num_factors - 1):
        seq.append("mul")
    if num_constants > 0:
        seq.append("mul")
        seq.append("C")
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if any(t in str(cur_fact) for t in tokens):
            seq = seq + from_sympy_to_seq(cur_fact)
    return seq


def from_sympy_add_to_seq(sympy_add_expr):
    """
    C.f. from_sympy_to_seq function.
    Standardize the sequence of tokens for additions.
    """
    tokens = ["x1", "x2", "x3", "x4", "x5", "x6"]
    num_terms = 0
    num_constants = 0
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            num_terms += 1
        else:
            num_constants += 1
    seq = []
    for _ in range(num_terms - 1):
        seq.append("add")
    if num_constants > 0:
        seq.append("add")
        seq.append("C")
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            seq = seq + from_sympy_to_seq(cur_term)
    return seq


def from_seq_to_tokens(seq_int: paddle.Tensor) -> List:
    """Convert the sequence of model results into sequence of tokens."""
    seq_tokens = []
    for n in range(len(seq_int)):
        if seq_int[n] >= 2:
            seq_tokens.append(vocab_library[seq_int[n] - 2])
    return seq_tokens


def from_tokens_to_seq(seq_tokens: List) -> paddle.Tensor:
    """Convert the sequence of tokens into sequence of model results."""
    seq_int = []
    for token in seq_tokens:
        seq_int.append(vocab_library.index(token) + 2)
    return paddle.to_tensor(seq_int, dtype=paddle.int64).unsqueeze(0)


def from_seq_to_zss_tree(seq_tokens: List):
    """
    Convert the sequence into zss tree. Refer to https://arxiv.org/abs/2206.10540.
    Note: also works with sequences that do not correspond to complete equation trees!
    """
    cur_token = seq_tokens[0]
    if cur_token in ["add", "mul"]:
        split_idx = find_split_idx(seq_tokens)
        if split_idx is None:
            tree = zss.Node(cur_token)
            if len(seq_tokens[1:]) > 0:
                left_kid = from_seq_to_zss_tree(seq_tokens[1:])
                tree.addkid(left_kid)
        else:
            tree = zss.Node(cur_token)
            left_kid = from_seq_to_zss_tree(seq_tokens[1 : split_idx + 1])
            tree.addkid(left_kid)
            if len(seq_tokens[split_idx + 1 :]) > 0:
                right_kid = from_seq_to_zss_tree(seq_tokens[split_idx + 1 :])
                tree.addkid(right_kid)
        return tree
    elif cur_token in ["sin", "cos", "log", "exp", "neg", "inv", "sqrt", "sq", "cb"]:
        tree = zss.Node(cur_token)
        if len(seq_tokens[1:]) > 0:
            kid = from_seq_to_zss_tree(seq_tokens[1:])
            tree.addkid(kid)
        return tree
    elif cur_token in ["C", "x1", "x2", "x3", "x4", "x5", "x6"]:
        leaf = zss.Node(cur_token)
        return leaf


def find_split_idx(seq_tokens):
    """
    Helper function for from_seq_to_zss_tree.
    Locates the split index for binary nodes.
    """
    split_idx = 0
    arity = 1
    while arity > 0 and split_idx + 1 < len(seq_tokens):
        split_idx += 1
        if seq_tokens[split_idx] in ["add", "mul"]:
            arity += 1
        elif seq_tokens[split_idx] in ["C", "x1", "x2", "x3", "x4", "x5", "x6"]:
            arity += -1
    if split_idx + 1 == len(seq_tokens):
        split_idx = None
    return split_idx


def simplify_output(
    out_tensor: paddle.Tensor,
    mode: Literal["sympy", "token", "tensor"],
) -> paddle.Tensor:
    """Convert the model output results into the corresponding form according to the 'mode' and simplify it."""
    out_tokens = from_seq_to_tokens(out_tensor)
    out_sympy = from_seq_to_sympy(out_tokens)
    out_sympy_reassign = reassign_variables(out_sympy)

    out_sympy_simplify = sympy.simplify(sympy.factor(out_sympy_reassign))
    if mode == "sympy":
        return out_sympy_simplify

    out_re_tokens = from_sympy_to_seq(out_sympy_simplify)
    if mode == "token":
        return out_re_tokens

    out_re_tensor = from_tokens_to_seq(out_re_tokens)
    return out_re_tensor


def reassign_variables(sympy_expr):
    """
    Counts the number of variables in the SymPy expression and assign firte variables first.
    Example: log(x3)+x5 becomes log(x1)+x2
    """
    tokens = ["x1", "x2", "x3", "x4", "x5", "x6"]
    sympy_str = str(sympy_expr)
    exist = []
    for t in tokens:
        exist.append(t in sympy_str)
    for idx_new, idx_old in enumerate(np.where(exist)[0]):
        sympy_str = sympy_str.replace(f"x{idx_old+1}", f"x{idx_new+1}")
    sympy_expr = sympy.sympify(sympy_str)
    return sympy_expr


def is_tree_complete(seq_indices):
    """Check whether a given sequence of tokens defines a complete symbolic expression."""
    arity = 1
    for n in seq_indices:
        n = n.item()
        if n == 0 or n == 1:
            continue
            print("Predict padding or <SOS>, which is bad...")
        cur_token = vocab_library[n - 2]
        if cur_token in ["add", "mul"]:
            arity = arity + 2 - 1
        elif cur_token in [
            "sin",
            "cos",
            "log",
            "exp",
            "neg",
            "inv",
            "sqrt",
            "sq",
            "cb",
        ]:
            arity = arity + 1 - 1
        elif cur_token in ["C", "x1", "x2", "x3", "x4", "x5", "x6"]:
            arity = arity + 0 - 1
    if arity == 0:
        return True
    else:
        return False


def compute_norm_zss_dist(pred: paddle.Tensor, label: paddle.Tensor) -> float:
    """Computes ZSS tree edit distance, normalized by the length of the ground label and is between [0, 1].

    Args:
        pred (paddle.Tensor): Best sequence as predicted by the model. Typically the result of 'Beam Search' with k=1.
        label (paddle.Tensor): Ground_truth (feed to the decoder, shifted right).

    Returns:
        float: ZSS distance.
    """
    label = from_seq_to_tokens(label)
    pred = from_seq_to_tokens(pred)
    tree_truth = from_seq_to_zss_tree(label)
    tree_pred = from_seq_to_zss_tree(pred)
    dist = zss.simple_distance(tree_truth, tree_pred)
    norm_dist = dist / float(len(label))
    norm_zss_dist = min(1.0, norm_dist)
    return norm_zss_dist


def count_var_num(sympy_expr):
    """
    Assumes that the variables are properly numbered, i.e. 'reassign_variables' has been applied.
    Returns the number of variables in the SymPy expression.
    """
    num_var = 0
    while f"x{num_var+1}" in str(sympy_expr):
        num_var += 1
    return num_var


def expr_tree_depth(sympy_expr):
    """
    Recursive function!
    Count the maximum depth for a given SymPy expression.
    """
    if len(sympy_expr.args) == 0:
        return 1
    elif len(sympy_expr.args) == 1:
        return 1 + expr_tree_depth(sympy_expr.args[0])
    else:
        max_depth = 0
        for a in sympy_expr.args:
            temp_depth = expr_tree_depth(a)
            if temp_depth > max_depth:
                max_depth = temp_depth
        return 1 + max_depth


def gen_expr(vocab):
    """
    Recursive function!
    Generate one expression using the tokens and their respective probabiities provided by 'vocab'.
    """
    weights = vocab[:, 1].astype("float32")
    probs = weights / np.sum(weights)
    N = len(vocab)
    expr = []
    rand_idx = np.random.choice(N, p=probs)
    cur_token = vocab[rand_idx, 0]
    cur_arity = int(vocab[rand_idx, 2])
    expr.append(cur_token)
    if cur_arity == 0:
        return expr
    else:
        if cur_token in ["sin", "cos"]:
            idx1 = np.where(vocab[:, 0] == "sin")[0][0]
            idx2 = np.where(vocab[:, 0] == "cos")[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        elif cur_token in ["log", "exp"]:
            idx1 = np.where(vocab[:, 0] == "log")[0][0]
            idx2 = np.where(vocab[:, 0] == "exp")[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        else:
            new_vocab = vocab
        if cur_arity == 1:
            child = gen_expr(new_vocab)
            return expr + child
        elif cur_arity == 2:
            child1 = gen_expr(new_vocab)
            child2 = gen_expr(new_vocab)
            return expr + child1 + child2


def gen_samples(sympy_expr, num_samples=200):
    """
    Sample from SymPy expression.
    Variables are first sampled using log-uniform distributions.
    """
    np_x = np.power(10.0, np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 6)))
    f = sympy.lambdify([x1, x2, x3, x4, x5, x6], sympy_expr)
    np_y = f(np_x[:, 0], np_x[:, 1], np_x[:, 2], np_x[:, 3], np_x[:, 4], np_x[:, 5])
    return np_y, np_x
