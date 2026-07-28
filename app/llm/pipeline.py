"""The four-step LLM playground pipeline backed by spaCy + Qwen2.5-0.5B-Instruct
(4-bit GGUF) run on CPU via llama.cpp.

Because the model is instruction-tuned, ``generate`` returns a real answer to the
input (via the Qwen chat template) rather than continuing the text. The
end-of-input is marked with Qwen's end-of-turn token ``<|im_end|>`` (the chat
model's equivalent of the end-of-text marker).

A single llama.cpp instance (loaded with ``embedding=True``) is created lazily
and cached for the process lifetime — it serves tokenization, per-token
embeddings and chat generation, so only one model copy sits in RAM. On a small
no-GPU VM everything runs on CPU. The GGUF file is downloaded from the
HuggingFace hub on first use if it is not already present (see warnings.log)."""
from __future__ import annotations

import os
import re
import string
import threading
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.config import get_settings

_lock = threading.Lock()
_spacy_nlp = None
# A single llama.cpp instance (loaded with embedding=True) serves tokenization,
# per-token embeddings AND chat generation — one model copy keeps RAM low.
_llm = None

# Qwen end-of-turn token — marks the end of the user input for the chat model.
IM_END = "<|im_end|>"


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        with _lock:
            if _spacy_nlp is None:
                import spacy

                # Keep the tagger + parser (they give POS/dep used to label token
                # clusters); only ner/lemmatizer are unnecessary here.
                _spacy_nlp = spacy.load(get_settings().spacy_model, disable=["ner", "lemmatizer"])
    return _spacy_nlp


def _pos_tags(words: list[str]) -> list[str]:
    """Universal POS tag per word, tagged IN ORDER so the tagger/parser see the
    original sentence context (far more accurate than tagging isolated tokens).

    The words already came from spaCy's own tokenizer (see ``tokenize``), so we
    feed them back verbatim via ``Doc(words=...)`` to guarantee 1:1 alignment
    instead of letting spaCy re-tokenize and risk a length mismatch.
    """
    from spacy.tokens import Doc

    nlp = _get_spacy()
    doc = Doc(nlp.vocab, words=words)
    for _, proc in nlp.pipeline:  # tok2vec → tagger → parser → attribute_ruler
        doc = proc(doc)
    return [tok.pos_ for tok in doc]


def _model_path() -> str:
    """Return the local GGUF path, downloading it from the HF hub if missing."""
    settings = get_settings()
    if os.path.exists(settings.gguf_model_path):
        return settings.gguf_model_path
    from huggingface_hub import hf_hub_download

    target_dir = os.path.dirname(settings.gguf_model_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=settings.gguf_repo_id,
        filename=settings.gguf_filename,
        local_dir=target_dir,
    )


def _get_llm():
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                from llama_cpp import Llama

                s = get_settings()
                # embedding=True enables per-token embeddings; the same instance
                # still serves chat generation, so we only hold one model copy.
                _llm = Llama(
                    model_path=_model_path(), n_ctx=s.llm_n_ctx,
                    n_threads=s.llm_n_threads, embedding=True, verbose=False,
                )
    return _llm


# ─── Step 1: tokenize (spaCy linguistic tokens + end-of-input marker) ──────────


def tokenize(prompt: str) -> list[str]:
    doc = _get_spacy()(prompt)
    tokens = [tok.text_with_ws for tok in doc if tok.text.strip() != ""]
    tokens.append(IM_END)  # mark end of input so the chat model returns a response
    return tokens


# ─── Step 2: encode (one representative Qwen vocab id per token) ───────────────


def encode(tokens: list[str]) -> list[int]:
    llm = _get_llm()
    eot = int(llm.token_eos())  # Qwen <|im_end|> id
    ids: list[int] = []
    for tok in tokens:
        if tok.strip() == IM_END:
            ids.append(eot)
            continue
        pieces = llm.tokenize(tok.encode("utf-8"), add_bos=False, special=False)
        ids.append(int(pieces[0]) if pieces else eot)
    return ids


# ─── Step 3: embed (real per-token embeddings → PCA 2D) ───────────────────────


def _embed_matrix(llm, tokens: list[str]) -> np.ndarray:
    """One mean-pooled embedding vector per token."""
    vectors = []
    for tok in tokens:
        text = tok if tok.strip() else " "
        vec = np.asarray(llm.embed(text), dtype=float)
        if vec.ndim == 2:  # (n_subtokens, dim) → mean-pool to one vector per token
            vec = vec.mean(axis=0)
        vectors.append(vec)
    return np.vstack(vectors) if vectors else np.zeros((0, 0))


def _pca_points(matrix: np.ndarray, labels: list[str]) -> list[dict]:
    """Project embedding vectors to 2D and pair each with its token label."""
    if len(matrix) >= 3:
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    elif len(matrix) == 2:
        coords = matrix[:, :2]
    else:
        coords = np.zeros((len(matrix), 2))
    return [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": labels[i]}
        for i in range(len(labels))
    ]


def _dedup_tokens(tokens: list[str]) -> list[str]:
    """Collapse repeated tokens (by surface form, first occurrence wins) so the
    embedding space, its PCA plot, and the clustering aren't skewed by how often a
    token happens to appear — a prompt with five 'the's shouldn't get a 'the'
    cluster by sheer frequency. Case-sensitive: 'the' and 'The' stay distinct."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        key = tok.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


def embed(tokens: list[str], ids: list[int]) -> list[dict]:
    llm = _get_llm()
    tokens = _dedup_tokens(tokens)  # one point per distinct token (frequency-independent)
    labels = [tok.strip() or tok for tok in tokens]
    return _pca_points(_embed_matrix(llm, tokens), labels)


# ─── Step 4: generate (Qwen chat completion → a real response) ────────────────


def generate(prompt: str, max_new_tokens: int = 96) -> dict:
    """Return the generated response as per-token chips, their vocab ids, and a
    2D PCA projection of the output token embeddings (symmetric with the input
    embedding step)."""
    llm = _get_llm()
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    text = (result["choices"][0]["message"]["content"] or "").strip()

    # Split the answer into per-token chips (with their ids) for the UI. The chips
    # keep every token (they show the real generated sequence), but the embedding
    # plot / clustering use distinct tokens only, so repeats don't bias them.
    token_ids = [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]
    tokens = [llm.detokenize([tid]).decode("utf-8", "replace") for tid in token_ids]
    uniq = _dedup_tokens(tokens)
    labels = [tok.strip() or tok for tok in uniq]
    points = _pca_points(_embed_matrix(llm, uniq), labels)

    return {"tokens": tokens, "ids": token_ids, "points": points}


# ─── K-means grouping of the embedding points + Qwen explanations ─────────────


CLUSTER_CAP = 8  # readability ceiling on the number of groups


def _pick_k(coords: np.ndarray, margin: float = 0.05) -> int:
    """Choose the cluster count, aiming for ~one group per 3 tokens.

    Silhouette almost always peaks at k=2 (one big word blob + a tight
    punctuation cluster), so optimising it alone collapses everything into two
    groups. Instead we make ~n/3 (capped at ``CLUSTER_CAP``) a *floor*, and let
    silhouette pick within ``[n/3, n/2]``: among all candidate k whose silhouette
    is within ``margin`` of the best, take the largest. This yields the finer,
    still reasonably-separated groups the playground wants.
    """
    n = len(coords)
    if n < 3:
        return 1
    # Floor at ~one cluster per 3 tokens; allow up to ~n/2. Both capped by the
    # point count and the readability ceiling.
    lo = min(max(2, round(n / 3)), n - 1, CLUSTER_CAP)
    hi = min(max(lo, round(n / 2)), n - 1, CLUSTER_CAP)

    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)
        if len(set(km.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(coords, km.labels_))
    if not scores:
        return min(lo, n - 1)

    best = max(scores.values())
    return max(k for k, s in scores.items() if s >= best - margin)


MIN_CLUSTER_SIZE = 2  # no singleton groups; undersized clusters merge into a neighbour


def _enforce_min_size(coords: np.ndarray, raw: list[int], min_size: int = MIN_CLUSTER_SIZE) -> list[int]:
    """Merge any cluster smaller than ``min_size`` into its nearest neighbour.

    Repeatedly takes the smallest undersized cluster and reassigns each of its
    points to the nearest *other* cluster centroid, until every remaining cluster
    meets the minimum (or only one cluster is left). Keeps granularity elsewhere —
    only the tiny clusters are dissolved.
    """
    raw = list(raw)
    while True:
        counts = Counter(raw)
        if len(counts) <= 1:
            break
        undersized = [lab for lab, c in counts.items() if c < min_size]
        if not undersized:
            break
        target = min(undersized, key=lambda lab: counts[lab])
        others = [lab for lab in counts if lab != target]
        centroids = {lab: coords[[i for i, r in enumerate(raw) if r == lab]].mean(axis=0) for lab in others}
        for i, r in enumerate(raw):
            if r == target:
                raw[i] = min(others, key=lambda lab: float(np.linalg.norm(coords[i] - centroids[lab])))
    return raw


CATCH_ALL = "a catch-all group"

# Closed-class / function POS tags. These tokens cluster by grammatical role, not
# meaning, so they're pulled out of the semantic content-word clustering and put in
# a single "function words" bucket instead (see `cluster`).
_FUNCTION_POS = frozenset({"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"})

# Human labels for the Universal POS tags spaCy assigns. CCONJ/SCONJ are merged.
_POS_LABEL = {
    "NOUN": "common nouns",
    "PROPN": "proper nouns and names",
    "VERB": "action verbs",
    "AUX": "auxiliary verbs",
    "ADJ": "descriptive adjectives",
    "ADV": "adverbs",
    "ADP": "prepositions",
    "CONJ": "conjunctions and connectives",  # merged CCONJ + SCONJ
    "DET": "articles and determiners",
    "PRON": "pronouns",
    "NUM": "numbers",
    "PART": "particles",
    "INTJ": "interjections",
    "PUNCT": "punctuation and symbols",
    "SYM": "punctuation and symbols",
}
# Open-class tags carry meaning worth a richer semantic label ("types of animals"
# rather than just "common nouns"), so for these we still consult the model and
# fall back to the POS label only if it returns junk. Closed-class tags are
# already their own best label, so we use them verbatim without asking the model.
_OPEN_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def _pos_group_label(pos_tags: list[str]) -> tuple[str | None, bool]:
    """From a group's POS tags, return ``(label, is_open_class)`` for the tag a
    strict majority of the tokens share, or ``(None, False)`` if no tag does.

    Requiring a strict majority (not just a plurality) keeps genuinely mixed
    groups — no single dominant part of speech — out of the deterministic path so
    they fall through to the model / catch-all.
    """
    tags = ["CONJ" if t in ("CCONJ", "SCONJ") else t for t in pos_tags if t]
    if not tags:
        return None, False
    top = max(set(tags), key=tags.count)
    if tags.count(top) <= len(tags) / 2:
        return None, False
    return _POS_LABEL.get(top), (top in _OPEN_POS)


# WordNet gives a real semantic label ("carnivore", "edible fruit") for a
# *coherent* group of nouns — but only when the group genuinely shares a specific
# category. Hypernyms this general are useless as labels, so we never return them.
_WN_TOO_GENERAL = frozenset(
    "entity physical_entity abstraction object whole thing matter substance group "
    "living_thing organism artifact artefact instrumentality instrumentation "
    "causal_agent psychological_feature attribute state part relation".split()
)
_wordnet = None


def _get_wordnet():
    """Lazily load the WordNet corpus, downloading it once if missing. Returns the
    corpus reader, or None if it can't be made available (then callers degrade to
    the model). Bundling the corpus at deploy time avoids the runtime download."""
    global _wordnet
    if _wordnet is None:
        with _lock:
            if _wordnet is None:
                try:
                    from nltk.corpus import wordnet as wn
                    wn.ensure_loaded()
                    _wordnet = wn
                except LookupError:
                    try:
                        import nltk
                        nltk.download("wordnet", quiet=True)
                        nltk.download("omw-1.4", quiet=True)
                        from nltk.corpus import wordnet as wn
                        wn.ensure_loaded()
                        _wordnet = wn
                    except Exception:
                        _wordnet = False  # give up; use the model instead
                except Exception:
                    _wordnet = False
    return _wordnet or None


def _wordnet_label(words: list[str]) -> str | None:
    """Return a specific shared hypernym for a coherent group of nouns, else None.

    Polysemy cuts both ways: we must consider secondary senses so 'orange' (whose
    first sense is the colour) still counts as a fruit, but doing so also surfaces
    junk senses ('fox'/'wolf'/'bear' all have a pejorative *person* sense). So we
    track two signals — coverage across each word's top few senses, and coverage
    via the single *first* (most common) sense — and require BOTH: a hypernym must
    be near-universal across the group AND supported by the first sense of at least
    half the words. 'carnivore'/'fruit' pass (first-sense-backed); the spurious
    'person' reading of a mixed animal cluster fails the first-sense test → None,
    and the group falls back to its POS label.
    """
    wn = _get_wordnet()
    if wn is None:
        return None
    nouns = [w.lower() for w in words if w.isalpha()]
    if len(nouns) < 2:
        return None

    covers: "Counter" = Counter()        # via any of the top-K senses
    covers_first: "Counter" = Counter()  # via the single most-common sense only
    depth: dict = {}
    n_with_senses = 0
    for w in nouns:
        senses = wn.synsets(w, pos=wn.NOUN)[:3]
        if not senses:
            continue
        n_with_senses += 1
        first_ancestors = set()
        for path in senses[0].hypernym_paths():
            first_ancestors.update(path)
        ancestors = set(first_ancestors)
        for sense in senses[1:]:
            for path in sense.hypernym_paths():
                ancestors.update(path)
        for syn in ancestors:
            covers[syn] += 1
            if syn not in depth:
                depth[syn] = min(len(p) for p in syn.hypernym_paths())
        for syn in first_ancestors:
            covers_first[syn] += 1

    if n_with_senses < 2:
        return None
    # Near-universal coverage (allow one outlier in larger groups) + first-sense
    # backing from at least half the words.
    need = n_with_senses if n_with_senses < 5 else n_with_senses - 1
    need_first = max(2, (need + 1) // 2)
    cands = [s for s, c in covers.items()
             if c >= need and covers_first[s] >= need_first
             and depth[s] >= 4 and s.name().split(".")[0] not in _WN_TOO_GENERAL]
    if not cands:
        return None
    # Most-covering first (so 'furniture' beats 'table' for table/chair/desk), then
    # best first-sense support, then deepest/most-specific.
    best = max(cands, key=lambda s: (covers[s], covers_first[s], depth[s]))
    return best.lemma_names()[0].replace("_", " ")


def _fallback_label(tokens: list[str]) -> str:
    """Generic label used when the model echoes the tokens or returns junk."""
    stripped = [t.strip() for t in tokens if t.strip()]
    if stripped:
        punct = sum(1 for t in stripped if all(c in string.punctuation for c in t))
        if punct / len(stripped) >= 0.5:
            return "punctuation and symbols"
    return CATCH_ALL


# Framing words a real description is built from — ignored when checking whether a
# description is just an echo of the group's tokens.
_DESC_FRAMING = frozenset(
    "words word terms term names about and or of the a an to for with related "
    "type types kind kinds thing things common general various".split()
)


def _clean_desc(text: str, tokens: list[str]) -> str:
    """Tidy a model description; fall back to a generic label if it's empty, too
    long, signals no clear theme, or mostly just repeats the group's own tokens."""
    text = (text or "").strip()
    if text:
        text = text.splitlines()[0].strip()
    for pref in ("the phrase is:", "the theme is:", "theme:", "answer:", "category:", "these are", "the words are", "the group contains"):
        if text.lower().startswith(pref):
            text = text[len(pref):].strip()
    # Strip a trailing parenthetical (the small model tends to append "(e.g. ...)"
    # or a list of the group's own words in parentheses) and any "e.g."/"such as"
    # example tail — keep only the leading generalized category.
    text = re.sub(r"\s*[\(\[].*$", "", text).strip()
    text = re.split(r"\b(?:e\.g\.|such as|like)\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # The prompt forbids commas, so a comma means the model started listing the
    # group's tokens after a good opening phrase ("music, Beatles, unite, …") —
    # keep only the phrase before the first comma.
    text = text.split(",")[0].strip()
    text = text.strip().strip('"').strip("'").strip().rstrip(".").strip()
    if not text or len(text) > 70:
        return _fallback_label(tokens)
    # A real theme is a phrase ("music and culture"); a bare single word left after
    # comma-truncation is almost always a leftover token ("prove") — reject it.
    # (WordNet's own one-word labels like "carnivore" never pass through here.)
    if " " not in text:
        return _fallback_label(tokens)
    # Model signalled no clear common category.
    if text.lower() in ("catch-all", "catchall", "catch all", "mixed", "mixed bag", "none", "no clear category", "unrelated"):
        return CATCH_ALL
    # Vague meta-answers that describe nothing concrete, including ones that only
    # parrot the prompt's meta-language ("semantic ...", "grammatical ...").
    low = text.lower()
    META = ("semantic", "grammatical", "category", "categories", "theme", "themes",
            "words", "tokens", "group", "related", "related words", "similar", "similar words")
    if low in META or low.startswith(("semantic", "grammatical", "category", "categories")):
        return _fallback_label(tokens)
    # Reject echoes: descriptions that are just a bare list of the group's tokens.
    # A real theme ("words about women and gender") legitimately reuses a token as
    # its subject, so it's only an echo when there are NO framing words to give it
    # structure and the remaining content words are overwhelmingly the tokens.
    token_words = {t.strip().lower().strip(string.punctuation) for t in tokens if t.strip()}
    token_words.discard("")
    desc_words = [w.strip(string.punctuation) for w in re.split(r"[,\s]+", low) if w.strip(string.punctuation)]
    has_framing = any(w in _DESC_FRAMING for w in desc_words)
    content = [w for w in desc_words if w not in _DESC_FRAMING]
    if not has_framing and content and sum(1 for w in content if w in token_words) / len(content) >= 0.6:
        return _fallback_label(tokens)
    return text


def _describe_group(llm, tokens: list[str], pos_tags: list[str] | None = None) -> str:
    """Label a token cluster. spaCy POS (tagged in sentence context, see
    ``_pos_tags``) is the primary signal: a strict closed-class majority
    (conjunctions, determiners, …) is authoritative and returned verbatim, while
    an open-class majority (nouns, verbs, …) still gets a shot at a richer
    semantic label from Qwen and falls back to the POS label if the model returns
    junk. Punctuation-dominant groups are handled directly."""
    stripped = [t.strip() for t in tokens if t.strip()]
    if stripped:
        punct = sum(1 for t in stripped if all(c in string.punctuation for c in t))
        if punct / len(stripped) >= 0.6:
            return "punctuation and symbols"
        # Small leftover mixing punctuation with words has no real theme.
        if 0 < punct < len(stripped) and len(stripped) <= 3:
            return CATCH_ALL

    # spaCy POS decides the grammatical type. Closed-class labels are already the
    # best answer, so use them without troubling the flaky small model.
    pos_label, open_class = _pos_group_label(pos_tags or [])
    if pos_label is not None and not open_class:
        return pos_label

    words = [t for t in stripped if not all(c in string.punctuation for c in t)]
    # For a coherent group of nouns, WordNet gives a concrete category ("carnivore",
    # "edible fruit") the tiny model can't reliably produce. Only nouns/proper nouns
    # have useful hypernyms, so gate on the POS-dominant type; incoherent groups get
    # None back and fall through to the model.
    if pos_label in ("common nouns", "proper nouns and names"):
        wn_label = _wordnet_label(words)
        if wn_label is not None:
            return wn_label

    shown = ", ".join(t.strip() or t for t in tokens)
    messages = [
            {"role": "system", "content": (
                "You are given a group of related words that were clustered together. In 3 to 6 "
                "words, describe the DOMINANT topic or meaning the words are mostly about. The group "
                "is often noisy — several words won't fit — but there is almost always a rough topic; "
                "name it and ignore the outliers. Prefer a meaning-based topic over a grammar-based "
                "one. Reply 'mixed' ONLY if there is truly no common topic at all. "
                "Give ONE short phrase. Never list the given words, never repeat more than one of "
                "them, and never use commas. Do not use the words 'semantic', 'grammatical', or "
                "'category'. Reply with only the short phrase."
            )},
            {"role": "user", "content": "Words: woman, female, human, gender, individual, age"},
            {"role": "assistant", "content": "words about women and gender"},
            {"role": "user", "content": "Words: music, history, culture, world, fashion, art, influence"},
            {"role": "assistant", "content": "music and culture"},
            {"role": "user", "content": "Words: Beatles, band, songs, fans, popular, unite, inspire"},
            {"role": "assistant", "content": "the Beatles and their fans"},
            {"role": "user", "content": "Words: influential, cultural, fans, generations, audiences, influence, fashion"},
            {"role": "assistant", "content": "cultural influence and fame"},
            {"role": "user", "content": "Words: mother, father, sister, uncle, cousin"},
            {"role": "assistant", "content": "family relationship terms"},
            {"role": "user", "content": "Words: happy, sad, angry, afraid, proud"},
            {"role": "assistant", "content": "emotion words"},
            {"role": "user", "content": "Words: of, to, in, and, or, the"},
            {"role": "assistant", "content": "linking and function words"},
            # No common topic at all → 'mixed'.
            {"role": "user", "content": "Words: garden, quickly, seventeen, beneath, orange"},
            {"role": "assistant", "content": "mixed"},
            {"role": "user", "content": f"Words: {shown}"},
    ]
    # Self-consistency: the 0.5B model is noisy, so sample up to 3 times and take
    # the majority cleaned answer. This suppresses one-off junk ("prove and Beatles")
    # and one-off "mixed", and confirms a real theme when it recurs. Early-exit once
    # two runs agree on a non-fallback answer.
    votes: list[str] = []
    for _ in range(3):
        result = llm.create_chat_completion(messages=messages, max_tokens=24, temperature=0.3, top_p=0.9)
        votes.append(_clean_desc(result["choices"][0]["message"]["content"], tokens))
        if len(votes) >= 2 and votes[-1] == votes[-2] and votes[-1] != CATCH_ALL:
            break
    counts = Counter(votes)
    # Most-voted answer; ties broken toward a real theme over the catch-all.
    desc = max(votes, key=lambda v: (counts[v], v != CATCH_ALL))
    # If the model gave up (catch-all) but POS found a dominant open class, prefer
    # the concrete POS label ("common nouns", "action verbs") over a vague catch-all.
    if desc == CATCH_ALL and pos_label is not None:
        return pos_label
    return desc


def _standardize(coords: np.ndarray) -> np.ndarray:
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std[std == 0] = 1.0
    return (coords - mean) / std


def cluster(points: list[dict], n_clusters: int | None = None) -> dict:
    """Cluster the CONTENT words semantically; bucket function words & punctuation.

    Qwen's token embeddings cluster by grammatical role, so mixing function words
    and punctuation into k-means produces syntactic groups ("the, of, and, …").
    Instead we run k-means only on the content words (nouns, verbs, adjectives,
    adverbs, proper nouns) — where a semantic split is possible — and drop every
    function word into one "function words" bucket and all punctuation into one
    "punctuation and symbols" bucket. Clustering uses the 2D PCA coordinates shown
    in the scatter plot. The end-of-input marker is excluded; content groups are
    labelled by WordNet + Qwen. Returns {"n_groups", "groups":[...], "assignments"}.
    """
    full_assign = [-1] * len(points)  # per input point; -1 = excluded marker
    kept_idx = [i for i, p in enumerate(points) if p.get("label", "").strip() != IM_END]
    if not kept_idx:
        return {"n_groups": 0, "groups": [], "assignments": full_assign}

    labels = [points[i]["label"] for i in kept_idx]
    # POS-tag every kept token ONCE, in original order, so spaCy sees the real
    # sentence context; this drives both the content/function split and labelling.
    pos_tags = _pos_tags(labels)

    # Partition kept tokens (by local index) into content / function / punctuation.
    content, function, punct = [], [], []
    for li, (lab, pos) in enumerate(zip(labels, pos_tags)):
        s = lab.strip()
        if pos in ("PUNCT", "SYM") or (s and all(c in string.punctuation for c in s)):
            punct.append(li)
        elif pos in _FUNCTION_POS:
            function.append(li)
        else:
            content.append(li)

    llm = _get_llm()
    groups: list[dict] = []
    next_gid = 1

    # ── Semantic clustering of the content words only ──────────────────────────
    if content:
        ccoords = _standardize(np.array([[points[kept_idx[li]]["x"], points[kept_idx[li]]["y"]]
                                         for li in content], dtype=float))
        m = len(ccoords)
        if n_clusters is None:
            k = _pick_k(ccoords)
        else:
            k = max(1, min(int(n_clusters), m))
        if k <= 1:
            raw = [0] * m
        else:
            raw = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(ccoords).tolist()
            raw = _enforce_min_size(ccoords, raw)  # no singleton content groups
        for raw_lab in sorted(set(raw)):
            member_local = [content[j] for j in range(m) if raw[j] == raw_lab]
            toks = [labels[li] for li in member_local]
            pos_ = [pos_tags[li] for li in member_local]
            gid = next_gid
            next_gid += 1
            for li in member_local:
                full_assign[kept_idx[li]] = gid
            groups.append({"id": gid, "tokens": toks,
                           "description": _describe_group(llm, toks, pos_)})

    # ── Fixed buckets for the syntactic tokens ─────────────────────────────────
    for bucket, label in ((function, "function words"), (punct, "punctuation and symbols")):
        if not bucket:
            continue
        gid = next_gid
        next_gid += 1
        for li in bucket:
            full_assign[kept_idx[li]] = gid
        groups.append({"id": gid, "tokens": [labels[li] for li in bucket], "description": label})

    return {"n_groups": len(groups), "groups": groups, "assignments": full_assign}
