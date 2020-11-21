import csv
from typing import Dict, List


def compare_docs(query_doc, docs, ans_field):
    ret = []
    ans_val = query_doc[ans_field]
    for doc in docs:
        doc_val = doc[ans_field]
        if doc_val == ans_val:
            ret.append(True)
        else:
            ret.append(False)
    return ret


def get_first_relevant_rank(case):
    return case.index(True) + 1 if True in case else -1


def print_cases(csv_path, cases: Dict[str, List[bool]]):
    with open(csv_path, mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "FIRST_RELEVANT"])
        writer.writeheader()
        for case_id, rels in cases.items():
            first_rel = get_first_relevant_rank(rels)
            writer.writerow({"ID": case_id, "FIRST_RELEVANT": first_rel})


def print_summary(txt_path, cases, ks):
    with open(txt_path, mode="w") as f:
        for k in ks:
            f.write(f"Found at {k} = {found_at_k(cases, k):.3f}\n")
        f.write(f"MRR = {calc_mrr(cases):.3f}\n")
        f.write(f"MAP = {calc_map(cases):.3f}\n")


def found_at_k(cases, k):
    found = 0
    for case in cases:
        if get_first_relevant_rank(case) <= k:
            found += 1
    return found / len(cases)


def calc_rr(case):
    frr = get_first_relevant_rank(case)
    if frr > 0:
        return 1 / frr
    else:
        return 0


def calc_mrr(cases):
    mrr = 0
    for case in cases:
        mrr += calc_rr(case)
    return mrr / len(cases)


def calc_ap(case):
    precs = []
    for i, elm in enumerate(case):
        if elm:
            found_answers = len(precs) + 1
            precs.append(found_answers / (i + 1))
    return sum(precs) / len(precs) if len(precs) > 0 else 0


def calc_map(cases):
    map_val = 0
    for case in cases:
        map_val += calc_ap(case)
    return map_val / len(cases)
