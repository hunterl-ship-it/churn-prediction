"""Intervention matcher — re-exports from woodwide core."""

from woodwide.core import (
    GENERIC_INTERVENTION_CATALOG,
    INTERVENTION_TEMPLATE_CATALOGS,
    INTERVENTION_TEMPLATE_LABELS,
    build_interventions,
    current_intervention_catalog,
    dataframe_to_intervention_catalog,
    default_intervention_catalog,
    detect_intervention_template,
    intervention_catalog_to_dataframe,
    intervention_template_label,
    resolve_intervention_template,
    set_intervention_catalog_template,
)

__all__ = [
    "GENERIC_INTERVENTION_CATALOG",
    "INTERVENTION_TEMPLATE_CATALOGS",
    "INTERVENTION_TEMPLATE_LABELS",
    "build_interventions",
    "current_intervention_catalog",
    "dataframe_to_intervention_catalog",
    "default_intervention_catalog",
    "detect_intervention_template",
    "intervention_catalog_to_dataframe",
    "intervention_template_label",
    "resolve_intervention_template",
    "set_intervention_catalog_template",
]
