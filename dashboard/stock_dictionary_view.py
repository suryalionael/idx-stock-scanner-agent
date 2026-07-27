"""Read-only dashboard view for the Stock Dictionary — the single source
of truth explaining every metric/indicator/score shown anywhere in the
dashboard.

STRICTLY READ-ONLY, same contract as every other reference view
(daily_movers_view.py, ai_lab_view.py, etc.): no writes, no live fetch. Only
reads stock_scanner/configs/dictionary/*.yaml via
dashboard.data_loader.load_stock_dictionary(). Explanations are never
hardcoded in this file — adding or editing a term is a YAML change, not a
dashboard code change.
"""
from __future__ import annotations

import streamlit as st

from dashboard.data_loader import load_stock_dictionary

_REFERENCE_ICONS = {"source_code": "💻", "documentation": "📄", "external": "🔗"}


def _matches_query(entry: dict, query: str, alias_index: dict[str, str]) -> bool:
    if not query:
        return True
    q = query.strip().lower()
    haystack = " ".join([
        entry.get("title", ""), entry.get("short_name", ""), entry.get("definition", ""),
    ]).lower()
    if q in haystack:
        return True
    # An alias match counts too — searching "Price to Book Value" should
    # find the PBV entry even though that phrase never appears in its title.
    return any(q in alias for alias, entry_id in alias_index.items() if entry_id == entry["id"])


def _render_entry(entry: dict) -> None:
    with st.expander(f"{entry.get('title', entry['id'])}  ·  _{entry.get('short_name', '')}_"):
        st.markdown(f"**Definition**\n\n{entry.get('definition', '—')}")

        if entry.get("formula"):
            st.markdown("**Formula**")
            st.code(entry["formula"], language=None)

        if entry.get("interpretation"):
            st.markdown(f"**Interpretation**\n\n{entry['interpretation']}")

        if entry.get("how_scanner_uses_it"):
            st.markdown(f"**How the Scanner Uses It**\n\n{entry['how_scanner_uses_it']}")

        if entry.get("example"):
            st.markdown(f"**Example**\n\n{entry['example']}")

        related = entry.get("related_terms") or []
        if related:
            st.markdown("**Related Terms:** " + ", ".join(f"`{r}`" for r in related))

        references = entry.get("references") or []
        if references:
            st.markdown("**References**")
            for ref in references:
                icon = _REFERENCE_ICONS.get(ref.get("type"), "•")
                st.caption(f"{icon} {ref.get('value', '')}")


def render_stock_dictionary_tab() -> None:
    """📖 Stock Dictionary tab — read-only. See module docstring."""
    st.markdown("### 📖 Stock Dictionary")
    st.caption(
        "The single source of truth for every metric, score, indicator, and term shown "
        "anywhere in this dashboard. Search by name or alias, or browse by category."
    )

    payload = load_stock_dictionary()
    entries = payload["entries"]
    categories = payload["categories"]
    alias_index = payload["alias_index"]

    if not entries:
        st.info("📭 The Stock Dictionary has no entries configured yet.")
        return

    col_search, col_category = st.columns([3, 2])
    with col_search:
        query = st.text_input(
            "🔍 Search", placeholder="e.g. PBV, RSI, Retail Ratio, Price to Book Value...",
            key="dict_search",
        )
    with col_category:
        category_options = ["All categories"] + [
            categories.get(cat_id, {}).get("display_name", cat_id)
            for cat_id in sorted(categories, key=lambda c: categories.get(c, {}).get("display_name", c))
        ]
        category_choice = st.selectbox("Category", options=category_options, key="dict_category")

    display_to_id = {meta.get("display_name", cid): cid for cid, meta in categories.items()}
    selected_category_id = display_to_id.get(category_choice)

    filtered = [
        e for e in entries
        if _matches_query(e, query, alias_index)
        and (selected_category_id is None or e.get("category") == selected_category_id)
    ]
    filtered.sort(key=lambda e: e.get("title", e["id"]).lower())

    st.caption(f"{len(filtered)} of {len(entries)} term(s) shown.")
    if not filtered:
        st.info("No terms match your search/filter.")
        return

    for entry in filtered:
        _render_entry(entry)
