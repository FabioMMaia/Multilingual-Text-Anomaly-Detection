"""
LLM-guided anomaly annotation pipeline.

Implements:
  - TASK_CONTEXT: per-dataset anomaly criterion definitions
  - LLMAnnotator: pluggable LLM backend (Gemini API, OpenAI API, local GGUF via llama-cpp)
  - select_samples: score-guided or random sample selection
  - run_llm_active_loop: full pipeline (DeepSVDD -> LLM annotation -> SetFit -> DeepSAD)

Backend options:
  - "groq"     : Groq API — FREE tier, no credit card, 14400 req/day, ~fast
                 Models: llama-3.3-70b-versatile, gemma2-9b-it, mixtral-8x7b-32768
                 pip install groq
                 Get free key at: https://console.groq.com
  - "gemini"   : Google Gemini 2.0 Flash Lite
                 pip install google-genai
                 Note: free tier may be unavailable in some regions (e.g. Brazil)
  - "openai"   : OpenAI gpt-4o-mini
                 pip install openai
  - "llamacpp" : Local quantized model via llama-cpp-python (CPU-friendly, no GPU needed)
                 pip install llama-cpp-python
                 Recommended model: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1GB)

Usage example:
  annotator = LLMAnnotator(backend="gemini", api_key="YOUR_KEY")
  results = annotator.annotate_batch(texts, dataset_name="told_br")
"""

import json
import re
import time
from tqdm import tqdm
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------------------------
# Task context — anomaly criterion per dataset
# ---------------------------------------------------------------------------

TASK_CONTEXT = {
    "told_br": {
        "description": (
            "Brazilian Portuguese social media texts collected for hate speech detection. "
            "The anomaly class is HATE SPEECH — texts that attack, demean, or discriminate "
            "against people based on identity characteristics. "
            "The normal class is ALL other speech, including casual profanity, strong opinions, "
            "and arguments that do not target identity groups."
        ),
        "normal_description": (
            "Any text that does NOT target people based on identity (race, gender, religion, "
            "sexual orientation, etc.). This includes casual profanity, rude language, "
            "strong insults between individuals, and heated arguments — as long as they do "
            "not discriminate against a group. Profanity alone is NOT hate speech."
        ),
        "anomaly_criterion": (
            "Texts that constitute HATE SPEECH: language that attacks, demeans, dehumanizes, "
            "or incites discrimination against people specifically because of their race, "
            "ethnicity, gender, sexual orientation, religion, nationality, or disability. "
            "IMPORTANT — do NOT flag as anomalous: (1) casual profanity without group targeting, "
            "(2) personal insults between individuals with no identity-based discrimination, "
            "(3) strong political opinions without dehumanizing any group, "
            "(4) crude humor not targeting identity groups. "
            "Only flag texts where the attack is clearly directed AT A GROUP based on who they are."
        ),
    },
    "tweets_hs": {
        "description": (
            "English tweets collected for hate speech detection."
        ),
        "normal_description": (
            "Regular tweets with no hate speech or discriminatory language."
        ),
        "anomaly_criterion": (
            "Tweets that contain hate speech or discriminatory language targeting individuals "
            "or groups based on identity characteristics, including coded or implicit forms."
        ),
    },
    "pt_tweets": {
        "description": (
            "Brazilian Portuguese tweets labeled for sentiment analysis. "
            "In this dataset, POSITIVE sentiment is the MINORITY class and is treated as anomalous "
            "by convention — not because positivity is inherently wrong, but because it is the "
            "less frequent class used as the anomaly target."
        ),
        "normal_description": (
            "Tweets expressing negative sentiment."
        ),
        "anomaly_criterion": (
            "Tweets expressing POSITIVE sentiment. "
            "This is a frequency-based anomaly criterion: positive tweets are the minority class "
            "in this dataset. Score as anomalous (1.0) any tweet that is clearly positive, "
            "and as normal (0.0) any tweet that is clearly negative. "
            "Ignore any general notion of 'unusual' or 'harmful' — the only criterion is sentiment polarity."
        ),
    },
    "tweeteval": {
        "description": (
            "English tweets labeled for sentiment analysis (negative / neutral / positive). "
            "In this dataset, NEUTRAL sentiment is the MAJORITY class and is treated as normal. "
            "Both NEGATIVE and POSITIVE sentiment are minority classes and are treated as anomalous."
        ),
        "normal_description": (
            "Tweets expressing neutral sentiment — neither clearly positive nor clearly negative."
        ),
        "anomaly_criterion": (
            "Tweets expressing STRONG sentiment in either direction: clearly NEGATIVE (criticism, anger, sadness, "
            "sarcasm, complaints) OR clearly POSITIVE (enthusiasm, praise, excitement, strong approval). "
            "This is a frequency-based anomaly: only neutral tweets are the majority (normal) class. "
            "Score as anomalous (1.0) any tweet with clear emotional polarity (positive or negative), "
            "and as normal (0.0) only tweets that are genuinely neutral or ambiguous."
        ),
    },
    "20_newsgroups": {
        "description": (
            "English newsgroup posts from 20 different topic categories. "
            "In this dataset, posts from the comp.graphics newsgroup are the MINORITY class "
            "and are treated as anomalous. Posts from all other 19 topics are the MAJORITY "
            "class and are treated as normal."
        ),
        "normal_description": (
            "Posts from ANY newsgroup EXCEPT comp.graphics: sports, hockey, religion, politics, "
            "science, medicine, automobiles, space, electronics, etc. "
            "The vast majority of posts in this dataset belong to these non-graphics topics."
        ),
        "anomaly_criterion": (
            "Posts that belong specifically to the comp.graphics newsgroup — discussions about "
            "computer graphics, image formats (GIF, JPEG, PNG), 3D rendering, raytracing, "
            "image processing, graphics software, or related technical topics. "
            "This is a frequency-based anomaly: comp.graphics posts are the rare minority class. "
            "Score as anomalous (1.0) any post clearly about computer graphics or image processing, "
            "and as normal (0.0) any post about any other topic (sports, politics, religion, etc.)."
        ),
    },
    "wikinews": {
        "description": (
            "Portuguese WikiNews articles from multiple topic sections. "
            "Your task is to classify whether an article belongs to the Politics (Política) section "
            "or to some other section (health, sports, science, culture, technology, environment). "
            "Politics = score 0.0. Any other section = score 1.0. "
            "IMPORTANT: The vast majority of articles you will see are from Politics — "
            "most articles should receive score 0.0. A score of 1.0 should be the exception, "
            "reserved only for articles whose primary topic is clearly non-political."
        ),
        "normal_description": (
            "News articles whose PRIMARY topic is political — ANY political topic from ANY country. "
            "This includes: elections, government decisions, coups, diplomacy, international relations, "
            "military conflicts and war reporting, political parties, legislation, political figures, "
            "press freedom, geopolitical crises, and economic policy decisions made by governments. "
            "The article does NOT need to be about Brazil or Portugal — politics worldwide counts as normal (0.0). "
            "An article about a health crisis, economy, or science that is primarily framed as a "
            "GOVERNMENT DECISION or POLITICAL ACTION is still Politics and scores 0.0. "
            "When in doubt, default to 0.0."
        ),
        "anomaly_criterion": (
            "News articles whose PRIMARY topic is clearly NOT political: "
            "pure health/medicine news (disease outbreaks, medical treatments, epidemics), "
            "pure sports results, pure culture/entertainment, pure science/technology discoveries, "
            "pure environment/agriculture topics. "
            "Score 1.0 ONLY when the article is unambiguously about one of these non-political domains "
            "AND contains no significant political angle. "
            "Elections, wars, diplomatic relations, court rulings on political matters, "
            "government policies — all of these are Politics (score 0.0), regardless of how dramatic they are. "
            "The fact that an event is important, historic, or controversial does NOT make it anomalous — "
            "score is based purely on topic section, not on newsworthiness."
        ),
    },
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are evaluating a text sample for anomaly detection.

Dataset context: {description}

Normal samples are: {normal_description}

Anomaly criterion (use this definition strictly):
{anomaly_criterion}

Rate the following text on a scale from 0.0 (clearly normal) to 1.0 (clearly anomalous \
according to the criterion above). Do not use any other notion of "anomalous".

Text:
\"\"\"
{text}
\"\"\"

Respond ONLY with a JSON object, no explanation outside the JSON:
{{"anomaly_score": <float between 0.0 and 1.0>, "reason": "<one sentence explaining your score>"}}
"""


def build_prompt(text: str, dataset_name: str) -> str:
    """Build a fully formatted annotation prompt for a given text and dataset."""
    if dataset_name not in TASK_CONTEXT:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(TASK_CONTEXT.keys())}"
        )
    ctx = TASK_CONTEXT[dataset_name]
    return PROMPT_TEMPLATE.format(text=text, **ctx)


def parse_llm_response(response_text: str) -> dict:
    """
    Parse LLM response into {'anomaly_score': float, 'reason': str}.
    Handles common formatting issues (markdown code blocks, extra text).
    Returns {'anomaly_score': None, 'reason': response_text} on failure.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    # Extract first JSON object
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return {"anomaly_score": None, "reason": response_text}
    try:
        parsed = json.loads(match.group())
        score = float(parsed.get("anomaly_score", -1))
        if not 0.0 <= score <= 1.0:
            return {"anomaly_score": None, "reason": response_text}
        return {"anomaly_score": score, "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, ValueError):
        return {"anomaly_score": None, "reason": response_text}


# ---------------------------------------------------------------------------
# LLM Annotator — pluggable backend
# ---------------------------------------------------------------------------

class LLMAnnotator:
    """
    LLM-based text annotator with pluggable backends.

    Backends:
        "gemini"   — Google Gemini 2.0 Flash via google-genai SDK
        "openai"   — OpenAI gpt-4o-mini via openai SDK
        "llamacpp" — Local GGUF model via llama-cpp-python (CPU inference)

    Args:
        backend: One of "gemini", "openai", "llamacpp".
        api_key: API key for gemini/openai (or set via env var).
        model: Model name/path override.
                gemini default  : "gemini-2.0-flash"
                openai default  : "gpt-4o-mini"
                llamacpp default: path must be provided explicitly
        llamacpp_kwargs: Extra kwargs passed to llama_cpp.Llama (e.g. n_ctx, n_threads).
        retry_delay: Seconds to wait between retries on API errors.
        max_retries: Max retries per sample.
    """

    BACKEND_DEFAULTS = {
        "groq"    : "llama-3.3-70b-versatile",  # free: 14400 req/day, 6000 TPM
        "gemini"  : "gemini-2.0-flash-lite",     # free tier may be geo-restricted
        "openai"  : "gpt-4o-mini",
        "llamacpp": None,                         # path must be provided
    }

    def __init__(
        self,
        backend: Literal["groq", "gemini", "openai", "llamacpp"] = "groq",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llamacpp_kwargs: Optional[dict] = None,
        retry_delay: float = 2.0,
        max_retries: int = 3,
    ):
        if backend not in self.BACKEND_DEFAULTS:
            raise ValueError(f"backend must be one of {list(self.BACKEND_DEFAULTS)}")

        self.backend = backend
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.model = model or self.BACKEND_DEFAULTS[backend]
        self._client = None

        # Lazy init — actual import happens here to avoid hard dependency
        if backend == "groq":
            self._init_groq(api_key)
        elif backend == "gemini":
            self._init_gemini(api_key)
        elif backend == "openai":
            self._init_openai(api_key)
        elif backend == "llamacpp":
            self._init_llamacpp(llamacpp_kwargs or {})

    # --- backend init ---

    @staticmethod
    def _load_dotenv():
        """Load .env from project root if python-dotenv is available."""
        try:
            from dotenv import load_dotenv, find_dotenv
            # find_dotenv() searches up the directory tree — works regardless of cwd
            dotenv_path = find_dotenv(usecwd=True)
            load_dotenv(dotenv_path, override=False)
        except ImportError:
            pass  # python-dotenv is optional

    def _init_groq(self, api_key):
        self._load_dotenv()
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: pip install groq")
        import os
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("Provide api_key or set GROQ_API_KEY env var (or add to .env)")
        self._client = Groq(api_key=key)

    def _init_gemini(self, api_key):
        self._load_dotenv()
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")
        import os
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Provide api_key or set GEMINI_API_KEY env var (or add to .env)")
        self._client = genai.Client(api_key=key)

    def _init_openai(self, api_key):
        self._load_dotenv()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Provide api_key or set OPENAI_API_KEY env var (or add to .env)")
        self._client = OpenAI(api_key=key)

    def _init_llamacpp(self, llamacpp_kwargs):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Install llama-cpp-python: pip install llama-cpp-python\n"
                "Recommended model: Qwen2.5-7B-Instruct-Q4_K_M.gguf (~4.7GB)\n"
                "Download from: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF"
            )
        if not self.model:
            raise ValueError("Provide model path for llamacpp backend, e.g. model='/path/to/model.gguf'")
        # n_gpu_layers=-1 offloads all layers to GPU; set 0 for CPU-only
        defaults = {"n_ctx": 8192, "n_threads": 4, "n_gpu_layers": -1, "verbose": False}
        defaults.update(llamacpp_kwargs)
        self._client = Llama(model_path=self.model, **defaults)

    # --- single sample annotation ---

    def _call_groq(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return response.text

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _truncate_for_llamacpp(self, text: str, dataset_name: str) -> tuple:
        """Truncate text so the full prompt fits within n_ctx - 300 tokens.
        Returns (text, truncated: bool).
        """
        max_prompt_tokens = self._client.n_ctx() - 300  # reserve 300 for response
        full_prompt = build_prompt(text, dataset_name)
        tokens = self._client.tokenize(full_prompt.encode())
        if len(tokens) <= max_prompt_tokens:
            return text, False
        # Measure overhead from the template (empty text)
        template_tokens = self._client.tokenize(build_prompt("", dataset_name).encode())
        text_budget = max_prompt_tokens - len(template_tokens)
        if text_budget <= 0:
            return text[:200], True  # safety fallback
        text_tokens = self._client.tokenize(text.encode())
        if len(text_tokens) > text_budget:
            text_tokens = text_tokens[:text_budget]
            text = self._client.detokenize(text_tokens).decode("utf-8", errors="replace")
        return text, True

    def _call_llamacpp(self, prompt: str) -> str:
        response = self._client.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()

    def annotate(self, text: str, dataset_name: str) -> dict:
        """
        Annotate a single text sample.

        Returns:
            dict with keys:
                'anomaly_score' : float in [0, 1] or None if parsing failed
                'reason'        : str (LLM's explanation)
                'parse_error'   : bool
        """
        truncated = False
        if self.backend == "llamacpp":
            text, truncated = self._truncate_for_llamacpp(text, dataset_name)
        prompt = build_prompt(text, dataset_name)
        raw = None
        for attempt in range(self.max_retries):
            try:
                if self.backend == "groq":
                    raw = self._call_groq(prompt)
                elif self.backend == "gemini":
                    raw = self._call_gemini(prompt)
                elif self.backend == "openai":
                    raw = self._call_openai(prompt)
                elif self.backend == "llamacpp":
                    raw = self._call_llamacpp(prompt)
                break
            except Exception as exc:
                warnings.warn(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {exc}")
                if attempt < self.max_retries - 1:
                    # Parse wait time from Groq 429 error messages
                    wait = self.retry_delay
                    exc_str = str(exc)
                    import re as _re
                    # Format 1: 'retryDelay': '17s'
                    m = _re.search(r"'retryDelay':\s*'([0-9.]+)s'", exc_str)
                    if m:
                        wait = float(m.group(1)) + 2.0
                    else:
                        # Format 2: "try again in 1m56.64s" or "try again in 47.3s"
                        m2 = _re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", exc_str)
                        if m2:
                            mins = float(m2.group(1) or 0)
                            secs = float(m2.group(2))
                            wait = mins * 60 + secs + 2.0
                    print(f"  [rate limit] waiting {wait:.0f}s before retry...", flush=True)
                    time.sleep(wait)
                else:
                    return {"anomaly_score": None, "reason": str(exc)[:200], "parse_error": True, "truncated": truncated}

        result = parse_llm_response(raw)
        result["parse_error"] = result["anomaly_score"] is None
        result["truncated"] = truncated
        return result

    def annotate_batch(
        self,
        texts: list,
        dataset_name: str,
        verbose: bool = True,
        delay: float = 0.0,
    ) -> list:
        """
        Annotate a list of texts. Returns list of dicts (same as annotate()).

        Args:
            texts: List of raw text strings.
            dataset_name: Key in TASK_CONTEXT.
            verbose: Print progress.
            delay: Seconds to wait between API calls (rate limiting).
        """
        results = []
        n = len(texts)
        for i, text in enumerate(texts):
            result = self.annotate(text, dataset_name)
            results.append(result)
            if verbose and (i + 1) % max(1, n // 4) == 0:
                n_errors = sum(1 for r in results if r["parse_error"])
                print(f"  [{i+1}/{n}] parse_errors={n_errors}", flush=True)
            if delay > 0:
                time.sleep(delay)
        if verbose:
            n_failed = sum(1 for r in results if r["parse_error"])
            print(f"  Done. {n - n_failed}/{n} parsed successfully.")
        return results


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(
    indices: np.ndarray,
    scores: np.ndarray,
    n: int,
    strategy: Literal["random", "score_guided"] = "score_guided",
    random_state: int = 42,
) -> np.ndarray:
    """
    Select N sample indices for LLM annotation.

    Args:
        indices: Array of available sample indices (into the full training pool).
        scores: Anomaly scores for each sample in `indices` (from unsupervised model).
        n: Number of samples to select.
        strategy:
            "score_guided" — select the n samples with the highest anomaly scores.
            "random"       — select n samples uniformly at random.
        random_state: Seed for random strategy.

    Returns:
        selected: Array of n indices.
    """
    n = min(n, len(indices))
    if strategy == "score_guided":
        order = np.argsort(scores)[::-1]  # highest scores first
        return indices[order[:n]]
    elif strategy == "random":
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(indices), size=n, replace=False)
        return indices[chosen]
    else:
        raise ValueError(f"strategy must be 'score_guided' or 'random', got '{strategy}'")


# ---------------------------------------------------------------------------
# Full active loop
# ---------------------------------------------------------------------------

def run_llm_active_loop(
    texts: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    annotator: LLMAnnotator,
    unsup_model_cls,
    semisup_model_cls,
    setfit_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    strategy: Literal["random", "score_guided"] = "score_guided",
    n_llm_calls: int = 100,
    anomaly_score_threshold: float = 0.6,
    min_anomalies_required: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Full LLM-guided anomaly detection pipeline (no human labels used in training).

    Pipeline:
        1. Train/test split (labels used ONLY for final evaluation).
        2. Run unsupervised model (e.g. DeepSVDD) on train embeddings -> anomaly scores.
        3. Select n_llm_calls samples (random or score-guided).
        4. Query LLM -> get anomaly_score per sample.
        5. Convert LLM scores to binary labels via threshold.
        6. Check: if fewer than min_anomalies_required labeled anomalies, warn and continue.
        7. Fine-tune SetFit on LLM-labeled samples.
        8. Re-encode train + test with SetFit embeddings.
        9. Train semi-supervised model (e.g. DeepSAD) on LLM labels + SetFit embeddings.
        10. Evaluate on test set -> return metrics.

    Args:
        texts: Raw text array (full dataset, pre-split).
        embeddings: Pre-computed embeddings (N x D).
        labels: Ground-truth binary labels (0=normal, 1=anomaly). Used ONLY for
                diagnostic metrics (LLM agreement, train AUC) — never for training.
        dataset_name: Key in TASK_CONTEXT (for prompt building).
        annotator: Initialized LLMAnnotator instance.
        unsup_model_cls: Class for the unsupervised AD model (e.g. deepod.models.DeepSVDD).
        semisup_model_cls: Class for the semi-supervised AD model (e.g. deepod.models.DeepSAD).
        setfit_model_name: SetFit base model for contrastive fine-tuning.
        strategy: Sample selection strategy ("random" or "score_guided").
        n_llm_calls: Budget of LLM annotation calls.
        anomaly_score_threshold: LLM scores >= threshold -> label as anomaly (1).
        min_anomalies_required: Minimum anomaly labels needed to proceed; warns if not met.
        test_size: Fraction of data held out for evaluation.
        random_state: Reproducibility seed.
        verbose: Print progress at each step.

    Returns:
        dict with keys:
            'roc_auc'           : float
            'pr_auc'            : float
            'n_llm_labeled'     : int (total LLM-labeled samples)
            'n_anomalies_found' : int (samples labeled as anomaly by LLM)
            'n_normals_found'   : int
            'n_parse_errors'    : int
            'strategy'          : str
            'n_llm_calls'       : int
            'dataset'           : str
            'llm_labels_df'     : pd.DataFrame (index, text, llm_score, llm_label, reason)
    """
    from setfit import SetFitModel, SetFitTrainer
    from datasets import Dataset as HFDataset

    # ------------------------------------------------------------------
    # Step 1 — Train / test split (labels not used in training)
    # ------------------------------------------------------------------
    indices = np.arange(len(texts))
    # We split indices only; ground-truth labels passed in for eval only
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    texts_train = texts[train_idx]
    texts_test = texts[test_idx]
    binary_labels = labels  # alias for clarity — used only in diagnostics

    if verbose:
        print(f"[1] Split: {len(train_idx)} train / {len(test_idx)} test")

    # ------------------------------------------------------------------
    # Step 2 — Unsupervised model -> anomaly scores
    # ------------------------------------------------------------------
    if verbose:
        print("[2] Training unsupervised model...")
    unsup_model = unsup_model_cls(random_state=random_state, device=device, verbose=0)
    unsup_model.fit(X_train)
    unsup_scores = unsup_model.decision_function(X_train)  # higher = more anomalous

    if verbose:
        print(f"    Scores range: [{unsup_scores.min():.4f}, {unsup_scores.max():.4f}]")

    # ------------------------------------------------------------------
    # Step 3 — Select samples for LLM annotation
    # ------------------------------------------------------------------
    if verbose:
        print(f"[3] Selecting {n_llm_calls} samples ({strategy})...")
    selected_local_idx = select_samples(
        indices=np.arange(len(train_idx)),
        scores=unsup_scores,
        n=n_llm_calls,
        strategy=strategy,
        random_state=random_state,
    )
    selected_texts = texts_train[selected_local_idx]

    # ------------------------------------------------------------------
    # Step 4 — LLM annotation
    # ------------------------------------------------------------------
    if verbose:
        print(f"[4] Querying LLM ({annotator.backend} / {annotator.model})...")
    llm_results = annotator.annotate_batch(selected_texts, dataset_name, verbose=verbose)

    # ------------------------------------------------------------------
    # Step 5 — Convert to binary labels
    # ------------------------------------------------------------------
    llm_scores = np.array([
        r["anomaly_score"] if r["anomaly_score"] is not None else -1.0
        for r in llm_results
    ])
    valid_mask = llm_scores >= 0  # exclude parse errors
    llm_labels = np.where(llm_scores >= anomaly_score_threshold, 1, 0)
    llm_labels[~valid_mask] = -1  # mark parse errors

    n_valid = valid_mask.sum()
    n_anomalies = (llm_labels[valid_mask] == 1).sum()
    n_normals = (llm_labels[valid_mask] == 0).sum()
    n_errors = (~valid_mask).sum()

    if verbose:
        print(f"    Valid: {n_valid} | Anomalies: {n_anomalies} | Normals: {n_normals} | Errors: {n_errors}")

    if n_anomalies < min_anomalies_required:
        warnings.warn(
            f"Only {n_anomalies} anomaly samples labeled by LLM "
            f"(min required: {min_anomalies_required}). "
            f"Results may be unreliable."
        )
    anomaly_rate = n_anomalies / max(1, n_valid)
    if anomaly_rate > 0.5:
        warnings.warn(
            f"LLM labeled {n_anomalies}/{n_valid} ({anomaly_rate:.0%}) as anomaly — suspiciously high. "
            f"The LLM may have ignored the prompt (check backend/model format)."
        )

    # Keep only valid-labeled samples for training
    train_mask = valid_mask
    sf_texts = selected_texts[train_mask].tolist()
    sf_labels = llm_labels[train_mask].tolist()

    # ------------------------------------------------------------------
    # Step 6 — SetFit fine-tuning on LLM labels
    # ------------------------------------------------------------------
    n_anomalies_sf = sf_labels.count(1)
    n_normals_sf = sf_labels.count(0)
    n_classes = len(set(sf_labels))
    min_per_class = 8  # SetFit paper recommends >= 8 shots per class for stable results

    setfit_skipped = n_classes < 2 or n_anomalies_sf < min_per_class or n_normals_sf < min_per_class

    if setfit_skipped:
        warnings.warn(
            f"SetFit skipped: LLM produced {n_anomalies_sf} anomalies and {n_normals_sf} normals "
            f"(need >= {min_per_class} per class). "
            f"Using original embeddings for DeepSAD. "
            f"Try increasing --n_llm_calls or using score_guided strategy."
        )
        X_train_sf = embeddings[train_idx]
        X_test_sf = embeddings[test_idx]
    else:
        # Balance for SetFit: subsample majority class so contrastive pairs are meaningful.
        # DeepSAD uses ALL labeled samples (not balanced) — handled below.
        n_per_class_setfit = min(n_anomalies_sf, n_normals_sf)
        rng = np.random.default_rng(random_state)

        anomaly_idx = [i for i, l in enumerate(sf_labels) if l == 1]
        normal_idx  = [i for i, l in enumerate(sf_labels) if l == 0]
        sel_anomaly = rng.choice(anomaly_idx, size=n_per_class_setfit, replace=False).tolist()
        sel_normal  = rng.choice(normal_idx,  size=n_per_class_setfit, replace=False).tolist()
        balanced_idx = sel_anomaly + sel_normal

        sf_texts_bal = [sf_texts[i] for i in balanced_idx]
        sf_labels_bal = [sf_labels[i] for i in balanced_idx]

        if verbose:
            print(f"[5] Fine-tuning SetFit: {n_per_class_setfit}×2 balanced samples "
                  f"(from {n_anomalies_sf} anomalies / {n_normals_sf} normals)...")

        setfit_model = SetFitModel.from_pretrained(setfit_model_name)
        train_dataset = HFDataset.from_dict({"text": sf_texts_bal, "label": sf_labels_bal})

        trainer = SetFitTrainer(
            model=setfit_model,
            train_dataset=train_dataset,
            num_iterations=10,
            num_epochs=1,
            batch_size=4,
            seed=random_state,
        )
        trainer.train()

        # ------------------------------------------------------------------
        # Step 7 — Re-encode with SetFit embeddings
        # ------------------------------------------------------------------
        if verbose:
            print("[6] Re-encoding with SetFit embeddings...")
        encode_fn = trainer.model.model_body.encode
        X_train_sf = encode_fn(texts_train.tolist(), show_progress_bar=False)
        X_test_sf = encode_fn(texts_test.tolist(), show_progress_bar=False)

    # Build supervised training set for DeepSAD — uses ALL valid LLM-labeled samples
    # (not the balanced subset used for SetFit — DeepSAD handles imbalance natively)
    labeled_local_idx = selected_local_idx[train_mask]
    X_sup = X_train_sf[labeled_local_idx]
    y_sup = np.array(sf_labels)  # all valid labels, may be imbalanced

    # ------------------------------------------------------------------
    # Step 8 — Semi-supervised AD model
    # ------------------------------------------------------------------
    if verbose:
        print(f"[7] Training semi-supervised model with {len(y_sup)} LLM labels...")
    semisup_model = semisup_model_cls(random_state=random_state, device=device, verbose=0)
    semisup_model.fit(X_sup, y_sup)

    # ------------------------------------------------------------------
    # Step 9 — Evaluate on test set (ground-truth labels used here only)
    # ------------------------------------------------------------------
    test_scores = semisup_model.decision_function(X_test_sf)
    train_scores = semisup_model.decision_function(X_train_sf)

    # ------------------------------------------------------------------
    # Diagnostic: LLM label agreement with ground truth
    # (ground truth used only for logging — not for training)
    # ------------------------------------------------------------------
    gt_all      = binary_labels[train_idx][selected_local_idx]     # all N (for DataFrame)
    gt_selected = gt_all[train_mask]                               # valid only (for metrics)
    llm_pred    = np.array(sf_labels)
    n_agree     = int((gt_selected == llm_pred).sum())
    n_disagree  = int((gt_selected != llm_pred).sum())
    llm_precision = float((llm_pred[gt_selected == 1] == 1).sum()) / max(1, int((llm_pred == 1).sum()))
    llm_recall    = float((llm_pred[gt_selected == 1] == 1).sum()) / max(1, int((gt_selected == 1).sum()))
    llm_agreement = n_agree / max(1, len(gt_selected))

    if verbose:
        print(f"    LLM vs ground truth: {n_agree}/{len(gt_selected)} correct "
              f"(agreement={llm_agreement:.1%}, precision={llm_precision:.1%}, recall={llm_recall:.1%})")

    llm_labels_df = pd.DataFrame({
        "dataset": dataset_name,
        "strategy": strategy,
        "n_llm_calls": n_llm_calls,
        "seed": random_state,
        "text": selected_texts,
        "ground_truth": gt_all,
        "llm_score": llm_scores,
        "llm_label": llm_labels,
        "reason": [r["reason"] for r in llm_results],
        "parse_error": [r["parse_error"] for r in llm_results],
        "truncated": [r.get("truncated", False) for r in llm_results],
    })

    if verbose:
        print("[8] Done. Returning test scores for evaluation.")

    return {
        "test_scores"     : test_scores,
        "train_scores"    : train_scores,
        "train_labels"    : binary_labels[train_idx],
        "X_test_sf"       : X_test_sf,
        "test_idx"        : test_idx,
        "n_llm_labeled"   : n_valid,
        "n_anomalies_found": int(n_anomalies),
        "n_normals_found" : int(n_normals),
        "n_parse_errors"  : int(n_errors),
        "setfit_skipped"  : setfit_skipped,
        "strategy"        : strategy,
        "n_llm_calls"     : n_llm_calls,
        "dataset"         : dataset_name,
        "llm_agreement"   : round(llm_agreement, 4),
        "llm_precision"   : round(llm_precision, 4),
        "llm_recall"      : round(llm_recall, 4),
        "llm_labels_df"   : llm_labels_df,
    }
