"""Original, label-free averaging over two equivalent option assignments."""
from dataclasses import replace

from decision_core import probabilities


def question_variants(question, orders=1):
    if orders not in (1, 2):
        raise ValueError("Supported option-order counts are one and two")
    if orders == 1 or question.kind == "noul":
        return (question,)
    return question, replace(question, names=question.names[::-1], descriptions=question.descriptions[::-1])


def combine_orders(question, variants, rows, temperature=1.0, method="logit_mean"):
    if not rows or len(rows) != len(variants):
        raise ValueError("Every option order needs one measured logit row")
    aligned = []
    for variant, row in zip(variants, rows):
        if set(variant.names) != set(question.names) or len(row) < len(question.names):
            raise ValueError("Option identities changed or scores are missing")
        aligned.append([row[variant.names.index(name)] for name in question.names])
    if method == "logit_mean":
        return probabilities([sum(values) / len(aligned) for values in zip(*aligned)], temperature)
    if method == "probability_mean":
        distributions = [probabilities(row, temperature) for row in aligned]
        return [sum(values) / len(distributions) for values in zip(*distributions)]
    raise ValueError("Unsupported option-order combination")
