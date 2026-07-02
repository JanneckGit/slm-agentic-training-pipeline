"""
data_pipeline/complexity_taxonomy.py
==============================
Single source of truth for the SQL complexity taxonomy used across the pipeline.

Two related but distinct concepts:
- CANONICAL_COMPLEXITIES: the full label vocabulary used to NORMALIZE raw/compound
  labels onto a closed set (incl. "CTEs", needed for correct compound resolution
  like "subqueries and CTEs").
- The WORKING set of classes the pipeline actually keeps is configured in
  config/pipeline_config.yaml under `complexity_classes` (CTEs intentionally
  excluded) and read via load_complexity_classes(). The filter decision is:
  keep iff normalize_complexity(label) in complexity_classes.
"""
import logging

logger = logging.getLogger(__name__)

# Full canonical vocabulary (Reihenfolge = Hierarchie für Compound-Auflösung).
# Enthält "CTEs" weiterhin, damit Labels wie "subqueries and CTEs" korrekt
# aufgelöst werden – ob CTEs behalten werden, entscheidet separat der
# complexity_classes-Filter aus der Config.
CANONICAL_COMPLEXITIES = [
    "CTEs",
    "window functions",
    "subqueries",
    "multiple_joins",
    "set operations",
    "single join",
    "aggregation",
    "basic SQL",
]

# Explizite 1:1-Aliase für Fälle, die Substring-Matching nicht abdeckt
# (z.B. "multiple joins" enthält "multiple_joins" nicht als Substring,
# weil Underscore vs. Space).
COMPLEXITY_ALIASES = {
    "multiple joins": "multiple_joins",
}


def normalize_complexity(label: str) -> str:
    """Mappt Rohlabels auf CANONICAL_COMPLEXITIES; Fallback: 'basic SQL'."""
    # Fehlende / Platzhalter-Werte → stiller Fallback (kein Warning,
    # weil das im Seed-Set massenhaft vorkommt)
    if not label or label == "unknown":
        return "basic SQL"
    if label in CANONICAL_COMPLEXITIES:
        return label
    if label in COMPLEXITY_ALIASES:
        return COMPLEXITY_ALIASES[label]
    # Compound-Label: jedes kanonische Tag als Substring suchen, das
    # erste Match in Hierarchie-Reihenfolge gewinnt.
    low = label.lower()
    for canon in CANONICAL_COMPLEXITIES:
        if canon.lower() in low:
            return canon
    logger.warning(f"Unknown complexity label '{label}' → fallback 'basic SQL'")
    return "basic SQL"


def load_complexity_classes(config: dict, config_path: str) -> list[str]:
    """
    Returns the working set of complexity classes from config['complexity_classes'].
    Hard-fail (KeyError) mit Nennung von Key UND Config-Datei, falls fehlend/leer.
    """
    classes = config.get("complexity_classes")
    if not classes:
        raise KeyError(
            f"Pflicht-Key 'complexity_classes' fehlt (oder ist leer) in der "
            f"Config-Datei '{config_path}'. Bitte eine top-level Liste "
            f"'complexity_classes:' mit den gültigen Klassen ergänzen "
            f"(7 Klassen, CTEs ausgeschlossen)."
        )
    return list(classes)
