"""Source-backed method index, distinct from executable architecture presets."""

from copy import deepcopy
from .upstream_models import UPSTREAM_MODELS


SOURCES = {
    "survey": {
        "title": "Deep learning for time series forecasting: a survey of recent advances",
        "venue": "Frontiers of Computer Science (2026)",
        "url": "https://link.springer.com/content/pdf/10.1007/s11704-025-50947-3.pdf",
        "pages": "13, Figure 2",
    },
    "no-champions": {
        "title": "There are no Champions in Supervised Long-Term Time Series Forecasting",
        "venue": "TMLR (January 2026)",
        "url": "https://arxiv.org/pdf/2502.14045v2",
        "pages": "3; additional evaluated baselines on 4",
    },
    "deformtime": {
        "title": "DeformTime: capturing variable dependencies with deformable attention for time series forecasting",
        "venue": "TMLR (April 2025)",
        "url": "https://arxiv.org/pdf/2406.07438v3",
        "pages": "7, Sections 3.6 and 4.1",
    },
}

# All 44 rows of the survey's Figure 2. Canonical spellings replace figure typos
# (Onefits, TimeLLM, Dlinear, Leddam, TimesKAN).
SURVEY_FAMILIES = {
    "Recurrent": ["DeepAR", "SegRNN", "WITRAN", "P-sLSTM", "LCESN"],
    "Convolutional": ["SCINet", "TimesNet", "MICN", "TVNet"],
    "Graph": ["FourierGNN", "CrossGNN", "T-PatchGNN"],
    "Transformer": ["PatchTST", "Crossformer", "CARD", "iTransformer", "Pathformer", "SAMformer", "Timer-XL"],
    "Language model": ["One Fits All", "TEMPO", "Time-LLM", "TimeCMA"],
    "MLP": ["TSMixer", "FreTS", "SOFTS", "TimeMixer", "AMD"],
    "Diffusion": ["TimeGrad", "D3VAE", "TSDiff", "LDT", "ARMD", "D3U"],
    "Other": ["FiLM", "DLinear", "LTBoost", "MSGNet", "LEDDAM", "TimeMachine", "TS-LIF", "KooNPro", "TimeKAN", "SimpleTM"],
}


def method_collection() -> dict:
    methods = {}

    def add(name: str, family: str, source: str, page: int) -> None:
        key = name.lower().replace(" ", "-").replace("/", "-")
        entry = methods.setdefault(name, {
            "id": key, "name": name, "family": family,
            "kind": "non-neural baseline" if name == "LTBoost" else "neural",
            "status": "reference", "preset_id": None, "sources": [],
            "notes": "Listed for research; training integration is not available yet.",
        })
        entry["sources"].append({"source_id": source, "page": page})

    for family, names in SURVEY_FAMILIES.items():
        for name in names:
            add(name, family, "survey", 13)
    for name, family in [
        ("DLinear", "Linear"), ("PatchTST", "Transformer"),
        ("TimeMixer", "MLP"), ("iTransformer", "Transformer"),
        ("TimeXer", "Transformer"), ("N-BEATS", "MLP"),
    ]:
        add(name, family, "no-champions", 3)
    for name, family in [("S-Mamba", "State space"), ("xLSTMTime", "Recurrent"),
                         ("ModernTCN", "Convolutional")]:
        add(name, family, "no-champions", 4)
    for name, family in [
        ("DeformTime", "Deformable attention"), ("LightTS", "MLP"),
        ("DLinear", "Linear"), ("Crossformer", "Transformer"),
        ("PatchTST", "Transformer"), ("iTransformer", "Transformer"),
        ("TimeMixer", "MLP"), ("ModernTCN", "Convolutional"),
        ("CycleNet", "MLP"), ("TimeXer", "Transformer"),
    ]:
        add(name, family, "deformtime", 7)

    for name, notes in {
        "DLinear": "Editable decomposition and temporal-projection blocks with a forecast head. Target-only by default; add features or mix in other layers. Inspired by DLinear, not an exact reproduction.",
        "PatchTST": "Editable patch embedding and temporal attention. Target-only by default; adding features creates joint multivariate patches. No window normalization, residual attention or pretraining.",
        "iTransformer": "Editable cross-variable attention projects variable tokens back to the time axis for further layers. All selected features feed the forecast head. No window normalization or calendar covariates.",
    }.items():
        methods[name].update(status="adaptation", preset_id=f"research-{name.lower()}", notes=notes)
    methods["DLinear"]["family"] = "Linear"
    methods["LTBoost"]["notes"] = "Linear regression with boosted-tree residuals; included for completeness, not a neural network."
    methods["DeformTime"]["notes"] = "Variable and temporal deformable attention with a GRU decoder. Reference only; the existing GRU block does not implement DeformTime."
    for key, name in UPSTREAM_MODELS.items():
        methods[name].update(
            status="adaptation", preset_id=f"tslib-{key}",
            notes="Pinned MIT-licensed TSLib model core inside an editable pipeline. History is left-padded to a multiple of 32; the same-length forecast feeds the playground head. No calendar covariates. This hybrid is not a published benchmark reproduction."
                  + (" TSLib's simplified TSMixer variant, not Google's original implementation." if key == "tsmixer" else ""),
        )
    return deepcopy({"sources": SOURCES, "methods": sorted(methods.values(), key=lambda item: item["name"].casefold())})
