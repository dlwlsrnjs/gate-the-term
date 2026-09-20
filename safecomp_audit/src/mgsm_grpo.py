"""GRPO on MGSM with optional Pivot-Augmented Contrast Injection (PACI).

This is the real-model test of the paper's proposed intervention. Everything the
bandit could only suggest has to survive here.

The problem, restated. On a language the policy is bad at, a group of G rollouts
agrees on correctness -- all wrong -- far more often than chance. By Proposition 1
the correctness term then contributes exactly zero to every advantage in that
group, and the update is spent on the format and length terms. Raising the
correctness weight provably changes nothing. The languages that need the signal
most receive the least of it.

The intervention. MGSM is parallel: the same problem exists in every language with
the SAME numeric answer. So for a prompt q_L whose group is degenerate we can
generate extra rollouts from the same policy under an AUGMENTED conditioning

    q'_L  =  q_L  +  "here is the same problem in English: <q_en>"

and import them into the group. Three properties matter:

  * The augmentation raises the chance of a correct rollout, so it supplies the
    contrast the group lacks.
  * It does NOT change the requested output language, so the imported rollouts are
    still in L. Importing English rollouts instead would teach the model to answer
    Bengali questions in English -- we measure language consistency precisely to
    show this failure mode is avoided.
  * The importance ratio is pi(o | q_L) / pi(o | q'_L): the SAME policy, the same
    language, the same task. It is O(1). This is the whole reason the intervention
    can work where a uniform or reference proposal cannot -- for those, the ratio
    on an off-distribution rollout collapses to zero and annihilates exactly the
    sample that carried the contrast.

The ratio is length-normalised (a per-token geometric mean). A raw sequence-level
ratio over ~300 tokens is numerically hopeless, and normalising is what makes the
floor delta interpretable: delta is a bound on how far a single rollout may be
discounted, per token.

Loss (one update per batch of rollouts, so no PPO clipping is needed):

    L = - (1/N) sum_j  isw_j * A_j * mean_token_logp(o_j | q_L)

Usage:
  mgsm_grpo.py <mode> <out_name> [steps] [langs_csv]
    mode: vanilla | paci
Writes models/<out_name>_adapter and logs progress to stdout.
"""
import json, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lang_script import is_consistent

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
G = 4                     # on-policy group size
# Rollouts drawn from the operator per degenerate group. This is the sample size of
# the Proposition 3 estimator, and at the default of 2 it is the noisiest setting
# that still works: the verification puts its relative error near 1.2, and contrast
# is restored in only about half of degenerate groups, so the gate does nothing at
# all on the rest. INJ_REF keeps larger budgets COMPARABLE to the runs we already
# have: injected rollouts are weighted by INJ_REF/G_INJ, which is exactly 1 at the
# default, so raising K turns the target term from a sum over accepted draws into
# the mean over all K draws at unchanged magnitude. Coverage and variance then move;
# the effective step size does not.
G_INJ = int(os.environ.get("MGSM_GINJ", "2"))
INJ_REF = 2.0
B_PROMPTS = 2             # prompts per optimiser step
MAX_NEW = 320
LR = 1e-5
IS_FLOOR, IS_CLIP = 0.5, 2.0     # bounds on the length-normalised ratio
TAU = 1e-12
# A language-consistency term is not decoration. Without it nothing in the reward
# refers to the language at all, and on a low-resource language the cheapest way to
# raise correctness is to stop using that language: in a first run Telugu accuracy
# went 0.02 -> 0.46 while Telugu script usage went 0.92 -> 0.02. The model did not
# get better at Telugu, it abandoned it. Closing that shortcut is what makes the
# correctness-contrast question askable at all. The term is also the only auxiliary
# one that genuinely VARIES within a group, so it competes for the gradient that
# correctness cannot carry -- which is the paper's mechanism, not a nuisance.
WEIGHTS = {"corr": 1.0, "lang": 0.5, "fmt": 0.2, "brev": 0.3}
# `corr` below is GATED on language: a correct answer in the wrong language earns
# no correctness credit. A soft penalty is not enough -- switching language buys
# +1.0 of correctness, so any language weight under 1.0 leaves the shortcut
# profitable, and a weight above 1.0 would make language matter more than solving
# the problem. Gating states the actual requirement ("solve it IN this language")
# and leaves correctness as the target component within in-language responses,
# which is the quantity whose within-group contrast this paper is about.
TARGET_WORDS = 110.0
NONLATIN = {"ru", "zh", "ja", "th", "bn", "te"}
# Held-out problems are the LAST 60 of each shuffled language list; training draws
# only from the first 190, so the eval set is never trained on.
N_TRAIN, N_EVAL = 190, 60

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
PIVOT_TMPL = ("{q}\n\n(The same problem in English, for reference: {en})\n"
              "Answer in the language of the question above.")

# The operator above needs an English rendering of the problem. MGSM supplies one,
# which is convenient and also the method's sharpest weakness: a technique that
# needs aligned data to help low-resource languages is circular, because the
# languages that most need it are the ones least likely to have it. Under
# MGSM_SELF_PIVOT the reference is the model's OWN translation instead, so the
# operator uses nothing outside the policy. It is the harder setting, not the
# easier one -- the translation can be wrong, and a wrong pivot is a wrong hint.
SELF_PIVOT = os.environ.get("MGSM_SELF_PIVOT", "0") == "1"
TRANS_TMPL = ("Translate the following problem into English. "
              "Output only the translation, nothing else.\n\n{q}")
_TRANS_CACHE = {}


def pivot_reference(model, tok, ex, ex_en, lg, i):
    """English text placed inside the operator's prompt.

    Cached per problem. The policy drifts during training, but re-translating on
    every visit would make the operator a moving target and spend a generation per
    step to do it, so the translation is taken once from the policy that first
    needed it."""
    if not SELF_PIVOT:
        return ex_en["q"]
    key = (lg, i)
    if key not in _TRANS_CACHE:
        enc = tok([chat(tok, TRANS_TMPL.format(q=ex["q"]))],
                  return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=192, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        _TRANS_CACHE[key] = tok.decode(out[0, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip()
    return _TRANS_CACHE[key]


def norm(x):
    if x is None:
        return None
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def extract(t):
    m = NUM.findall(t.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def components(text, gold, lang=None):
    """corr is the target; fmt and brev are auxiliary shaping terms.

    fmt must be LANGUAGE-NEUTRAL. An earlier version accepted the English strings
    "the answer is" / "final answer", which in a multilingual setting makes the
    format term reward answering in English: a Bengali answer phrased in Bengali
    scored 0 while an English one scored 1. Since the correctness term is degenerate
    on most groups, the update flows to exactly this term, and the first training run
    showed language consistency falling (Bengali 0.76 -> 0.52, Telugu 0.92 -> 0.60).
    That is our own thesis playing out, but it also confounds the intervention: a
    gain from injection could be "it restored correctness contrast" or merely "it
    counteracted a pull toward English". The markers below -- a boxed answer, the
    GSM8K delimiter, or a final line containing a number -- carry no language."""
    pred = norm(extract(text))
    corr = float(pred is not None and pred == norm(gold))
    lines = [l for l in text.strip().splitlines() if l.strip()]
    last_has_num = bool(lines) and bool(NUM.search(lines[-1]))
    fmt = float(("\\boxed" in text) or ("####" in text) or last_has_num)
    # decided exactly from Unicode code points, so the reward carries no classifier
    # error; only defined for the non-Latin scripts this study is restricted to
    lg = is_consistent(text, lang) if lang in NONLATIN else 1.0
    corr_raw, corr = corr, corr * lg
    nw = max(1, len(text.split()))
    brev = 1.0 - min(1.0, abs(nw - TARGET_WORDS) / TARGET_WORDS)
    return {"corr": corr, "lang": lg, "fmt": fmt, "brev": brev,
            # kept for reporting only; not a reward term
            "corr_raw": corr_raw}


def reward(comp):
    return sum(w * comp[c] for c, w in WEIGHTS.items())


def load_parallel(langs):
    """MGSM is parallel: index i is the same problem in every language."""
    per = {}
    for lg in langs + ["en"]:
        rows = []
        for line in open(ROOT / f"data/mgsm/mgsm_{lg}.tsv", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows.append({"q": p[0], "gold": p[1].strip()})
        per[lg] = rows
    idx = list(range(min(len(v) for v in per.values())))
    random.Random(0).shuffle(idx)
    train, ev = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_EVAL]
    return per, train, ev


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(model, tok, prompt, n, temp=1.0):
    enc = tok([prompt], return_tensors="pt").to(DEV)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                         temperature=temp, top_p=1.0, num_return_sequences=n,
                         pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    seqs = out[:, plen:]
    texts = tok.batch_decode(seqs, skip_special_tokens=True)
    return texts


def mean_token_logp(model, tok, prompt, completions, grad=False):
    """mean per-token log pi(completion | prompt), one sequence at a time.

    Returned per-token rather than summed so that (a) the loss is not dominated by
    long rollouts and (b) the importance ratio below is a per-token geometric mean,
    which is the only form that stays numerically sane over hundreds of tokens."""
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for c in completions:
            pi = tok(prompt, return_tensors="pt").input_ids.to(DEV)
            ci = tok(c, return_tensors="pt", add_special_tokens=False
                     ).input_ids.to(DEV)
            if ci.shape[1] == 0:
                outs.append(torch.tensor(0.0, device=DEV))
                continue
            ids = torch.cat([pi, ci], 1)
            logits = model(ids).logits[:, :-1]
            tgt = ids[:, 1:]
            # Fused cross-entropy instead of materialising a float32 log_softmax
            # over the full 152k vocabulary: the explicit version allocated
            # hundreds of MB per sequence and put the run into OOM as soon as it
            # shared a card. Same value, a fraction of the memory.
            lp = -F.cross_entropy(logits.transpose(1, 2).float(), tgt,
                                  reduction="none")
            outs.append(lp[:, pi.shape[1] - 1:].mean())
    return outs



def logp_backward(model, tok, prompt, completion, coef):
    """Forward AND backward for ONE completion; returns its detached mean per-token
    log-probability, or None for an empty completion.

    The per-sequence backward is the whole point. Computing every completion's
    log-probability first and calling backward once keeps one float32 logits tensor
    per sequence alive -- about 365 MB for a 600-token sequence at a 152k
    vocabulary -- and that is what pushed this trainer to 66-72 GB and made every
    seed run die of CUDA OOM once other jobs shared the card. Gradients accumulate
    across calls, so summing the per-sequence backwards is exactly the same update
    at a fraction of the peak memory."""
    pi = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    ci = tok(completion, return_tensors="pt",
             add_special_tokens=False).input_ids.to(DEV)
    if ci.shape[1] == 0:
        return None
    ids = torch.cat([pi, ci], 1)
    logits = model(ids).logits[:, :-1]
    tgt = ids[:, 1:]
    lp = -F.cross_entropy(logits.transpose(1, 2).float(), tgt, reduction="none")
    m = lp[:, pi.shape[1] - 1:].mean()
    (coef * m).backward()
    return float(m.detach())


EVAL_N = int(os.environ.get("MGSM_EVAL_N", "25"))
EVAL_BS = int(os.environ.get("MGSM_EVAL_BS", "16"))


@torch.no_grad()
def gen_greedy_batch(model, tok, prompts, bs=None):
    """Greedy-decode a list of prompts in batches.

    Evaluating one problem at a time is what made this the dominant cost: decoding
    is sequential in tokens, so a batch of 16 costs barely more wall-clock than a
    batch of 1 and cuts the eval by an order of magnitude. Left padding (set on the
    tokenizer) is required for batched decoder-only generation."""
    bs = bs or EVAL_BS
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(DEV)
        gen_ids = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen_ids[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    return outs


def evaluate(model, tok, per, ev_idx, langs, max_per_lang=None):
    """greedy accuracy and language consistency on held-out problems, batched"""
    max_per_lang = max_per_lang or EVAL_N
    model.eval()
    res = {}
    for lg in langs:
        idxs = list(ev_idx[:max_per_lang])
        prompts = [chat(tok, per[lg][i]["q"]) for i in idxs]
        texts = gen_greedy_batch(model, tok, prompts)
        acc = lc = 0
        per_problem = []
        for i, t in zip(idxs, texts):
            cc = components(t, per[lg][i]["gold"], lg)
            c = cc["corr"]
            acc += c
            ok = is_consistent(t, lg) if lg in NONLATIN else None
            if ok is not None:
                lc += ok
            per_problem.append({"i": int(i), "corr": int(c),
                                "corr_raw": int(cc["corr_raw"]),
                                "lang": None if ok is None else int(ok)})
        n = len(idxs)
        res[lg] = {"acc": round(acc / max(1, n), 4),
                   "lang_consistent": (round(lc / max(1, n), 4)
                                       if lg in NONLATIN else None),
                   "n": n,
                   # per-problem outcomes, so the two arms can be compared with a
                   # PAIRED test (same problems, greedy decoding) instead of only
                   # by their marginal rates
                   "per_problem": per_problem}
    model.train()
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
    out_name = sys.argv[2] if len(sys.argv) > 2 else f"mgsm_{mode}"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    langs = (sys.argv[4].split(",") if len(sys.argv) > 4
             else ["bn", "te", "th", "sw"])
    # Seeding is argv[5]. Without it every "seed" would be the same run: the prompt
    # stream comes from random.Random and the rollouts from torch's sampler, so both
    # have to be seeded or a multi-seed replication is three copies of one result.
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1234
    assert mode in ("vanilla", "paci", "cgh", "cgh2", "mixen")

    per, train_idx, ev_idx = load_parallel(langs)
    print(f"mgsm-grpo mode={mode} seed={seed} langs={langs} steps={steps} "
          f"G={G} G_INJ={G_INJ if mode=='paci' else 0} "
          f"train={len(train_idx)} eval={len(ev_idx)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg).to(DEV)
    model.print_trainable_parameters()
    model.config.use_cache = True      # needed by generate(); the backward
                                       # pass here is short enough that we do
                                       # not need gradient checkpointing
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    log = {"mode": mode, "langs": langs, "steps": steps, "seed": seed, "G": G,
           "G_inj": G_INJ if mode == "paci" else 0,
           "weights": WEIGHTS, "is_floor": IS_FLOOR, "is_clip": IS_CLIP,
           "self_pivot": SELF_PIVOT, "g_inj": G_INJ,
           "history": []}
    print("baseline eval (before any update):", flush=True)
    log["eval_start"] = evaluate(model, tok, per, ev_idx, langs)
    print("  " + json.dumps(log["eval_start"]), flush=True)

    t0 = time.time()
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        stats = {"deg": 0, "inj": 0, "rew": [], "corr": []}
        for _ in range(B_PROMPTS):
            lg = rng.choice(langs)
            i = rng.choice(train_idx)
            ex, ex_en = per[lg][i], per["en"][i]
            prompt = chat(tok, ex["q"])

            texts = gen(model, tok, prompt, G)
            comps = [components(t, ex["gold"], lg) for t in texts]
            rs = [reward(c) for c in comps]
            isw = [1.0] * len(texts)
            src = ["on"] * len(texts)

            # --- PACI: import contrast when the target component is degenerate ---
            corr_vals = [c["corr"] for c in comps]
            degenerate = max(corr_vals) - min(corr_vals) <= TAU
            stats["deg"] += int(degenerate)
            if mode == "cgh2" and degenerate:
                # Component-gated hybrid. Proposition 1 says only the DEGENERATE
                # component has zero signal, so only its term is gated: the other
                # components keep their group-relative contribution, which is what
                # the group-level variant (mode "cgh") throws away -- in the bandit
                # that cost it a quarter of the auxiliary objective for no reason.
                pprompt = chat(tok, PIVOT_TMPL.format(q=ex["q"], en=pivot_reference(model, tok, ex, ex_en, lg, i)))
                inj = gen(model, tok, pprompt, G_INJ)
                lp_q = mean_token_logp(model, tok, prompt, inj)
                lp_p = mean_token_logp(model, tok, pprompt, inj)
                base_val = corr_vals[0]
                for t, a, b in zip(inj, lp_q, lp_p):
                    c = components(t, ex["gold"], lg)
                    if c["corr"] == base_val:
                        continue                      # contrast not restored
                    ratio = float(torch.exp(a - b).clamp(1e-6, 1e6))
                    texts.append(t); comps.append(c); rs.append(reward(c))
                    isw.append(min(max(ratio, IS_FLOOR), IS_CLIP)
                               * (INJ_REF / G_INJ))
                    stats["inj"] += 1

                # non-target components: group-relative as usual
                aux_w = {k: v for k, v in WEIGHTS.items() if k != "corr"}
                ra = torch.tensor([sum(w * c[k] for k, w in aux_w.items())
                                   for c in comps], dtype=torch.float32)
                A = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-4)
                # target: absolute signal, and only where contrast was restored
                tv = [c["corr"] for c in comps]
                if 0 < sum(tv) < len(tv):
                    A = A + WEIGHTS["corr"] * torch.tensor(tv, dtype=torch.float32)
                scale = 1.0 / (G * B_PROMPTS)
                for j, t in enumerate(texts):
                    logp_backward(model, tok, prompt, t,
                                  -isw[j] * float(A[j]) * scale)
                stats["rew"].append(float(np.mean(rs)))
                stats["corr"].append(float(sum(tv) / len(tv)))
                continue

            if mode == "cgh" and degenerate:
                # Contrast-Gated Hybrid. Proposition 1 says the group-relative
                # advantage carries exactly zero information about the target here,
                # so we switch ESTIMATOR rather than trying to repair the advantage.
                # Injected rollouts that restore contrast get a full positive signal
                # (advantage +1, no baseline); nothing is pushed down, which is what
                # separates this from the off-group baseline a noise control refuted
                # and from PACI, where the injected rollout had to compete with the
                # group mean. If nothing clears the bar the prompt is skipped: the
                # group provably has no signal on the target.
                pprompt = chat(tok, PIVOT_TMPL.format(q=ex["q"], en=pivot_reference(model, tok, ex, ex_en, lg, i)))
                inj = gen(model, tok, pprompt, G_INJ)
                lp_q = mean_token_logp(model, tok, prompt, inj)
                lp_p = mean_token_logp(model, tok, pprompt, inj)
                base_val = corr_vals[0]
                acc_texts, acc_w = [], []
                for t, a, b in zip(inj, lp_q, lp_p):
                    c = components(t, ex["gold"], lg)
                    if c["corr"] == base_val:
                        continue
                    ratio = float(torch.exp(a - b).clamp(1e-6, 1e6))
                    acc_texts.append(t)
                    acc_w.append(min(max(ratio, IS_FLOOR), IS_CLIP))
                stats["inj"] += len(acc_texts)
                if not acc_texts:
                    stats["skip"] = stats.get("skip", 0) + 1
                    continue                      # no signal on the target here
                scale = 1.0 / (G * B_PROMPTS)
                for t, w in zip(acc_texts, acc_w):
                    logp_backward(model, tok, prompt, t, -w * 1.0 * scale)
                stats["rew"].append(float(np.mean(rs)))
                stats["corr"].append(float(sum(corr_vals) / len(corr_vals)))
                continue

            if mode == "paci" and degenerate:
                pprompt = chat(tok, PIVOT_TMPL.format(q=ex["q"], en=pivot_reference(model, tok, ex, ex_en, lg, i)))
                inj = gen(model, tok, pprompt, G_INJ)
                lp_q = mean_token_logp(model, tok, prompt, inj)
                lp_p = mean_token_logp(model, tok, pprompt, inj)
                # Admit an injected rollout ONLY if it restores contrast on the
                # target. The algorithm's stated purpose is to supply the contrast
                # the group lacks; one that agrees with the degenerate group
                # supplies none and still dilutes every advantage in the group.
                # On this operator roughly half of injected rollouts are of that
                # kind, so the previous unconditional version spent half its extra
                # generations making the update worse.
                base_val = corr_vals[0]
                kept = 0
                for t, a, b in zip(inj, lp_q, lp_p):
                    c = components(t, ex["gold"], lg)
                    if c["corr"] == base_val:
                        continue                      # contrast not restored
                    texts.append(t); comps.append(c); rs.append(reward(c))
                    # length-normalised ratio, then floored and clipped
                    ratio = float(torch.exp(a - b).clamp(1e-6, 1e6))
                    isw.append(min(max(ratio, IS_FLOOR), IS_CLIP))
                    src.append("inj")
                    kept += 1
                stats["inj"] += kept

            if mode == "mixen" and degenerate:
                # Information-matched control for the gated estimator: the SAME
                # trigger, the SAME conditioning operator and the SAME extra
                # generation budget as cgh2, but only the standard estimator. The
                # pivot rollouts form their own group under their own conditioning
                # and get an ordinary group-relative advantage; no term is gated,
                # no importance weight is applied, and the original group falls
                # through to the vanilla update below. If the gated arm's gain
                # were mere access to the parallel English reference, this arm
                # would match it.
                #
                # The card is shared with other jobs whose footprint fluctuates;
                # if the extra injection work OOMs, skip THIS injection (counted)
                # rather than killing the run -- the on-policy update below still
                # proceeds.
                try:
                    pprompt = chat(tok, PIVOT_TMPL.format(q=ex["q"], en=pivot_reference(model, tok, ex, ex_en, lg, i)))
                    inj = gen(model, tok, pprompt, G_INJ)
                    icomps = [components(t, ex["gold"], lg) for t in inj]
                    ir = torch.tensor([reward(c) for c in icomps], dtype=torch.float32)
                    stats["inj"] += len(inj)
                    iA = (ir - ir.mean()) / (ir.std(unbiased=False) + 1e-4)
                    iscale = 1.0 / (G * B_PROMPTS)
                    for j, t in enumerate(inj):
                        logp_backward(model, tok, pprompt, t, -float(iA[j]) * iscale)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    stats["oom"] = stats.get("oom", 0) + 1

            r = torch.tensor(rs, dtype=torch.float32)
            A = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
            # Normalise by the ON-POLICY group size, not the augmented one.
            # Dividing by len(texts) gave PACI a 4/6 smaller update on the rollouts
            # both arms share, so the arms did not have matched effective step
            # sizes and the comparison was confounded in vanilla's favour.
            scale = 1.0 / (G * B_PROMPTS)
            for j, t in enumerate(texts):
                logp_backward(model, tok, prompt, t,
                              -isw[j] * float(A[j]) * scale)

            stats["rew"].append(float(r.mean()))
            stats["corr"].append(float(sum(c["corr"] for c in comps) / len(comps)))

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if (step + 1) % 10 == 0:
            el = (time.time() - t0) / 60
            row = {"step": step + 1,
                   "reward": round(sum(stats["rew"]) / len(stats["rew"]), 4),
                   "corr": round(sum(stats["corr"]) / len(stats["corr"]), 4),
                   "degenerate_groups": stats["deg"], "injected": stats["inj"],
                   "min": round(el, 1)}
            log["history"].append(row)
            print(f"  {json.dumps(row)}", flush=True)
        if (step + 1) % 50 == 0:
            model.save_pretrained(str(ROOT / f"models/{out_name}_adapter"))
            (ROOT / f"results/eval/mgsm_grpo_{out_name}.json").write_text(
                json.dumps(log, indent=2))

    print("final eval:", flush=True)
    log["eval_end"] = evaluate(model, tok, per, ev_idx, langs)
    print("  " + json.dumps(log["eval_end"]), flush=True)
    model.save_pretrained(str(ROOT / f"models/{out_name}_adapter"))
    (ROOT / f"results/eval/mgsm_grpo_{out_name}.json").write_text(
        json.dumps(log, indent=2))
    print(f"DONE mgsm_grpo mode={mode} -> models/{out_name}_adapter", flush=True)


if __name__ == "__main__":
    main()
