"""Rank hub traits and generate hotspot briefs for case study 2."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import networkx as nx
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
NETWORK_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "network"
)
HOTSPOT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "hotspots"
)
NOTES_DIR = (
    PROJECT_ROOT
    / ".notes"
    / "analysis-notes"
    / "case-study-analysis"
    / "cs2-hotspots"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Validate inputs without producing outputs",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )

    # ---- --network-dir ----
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=NETWORK_DIR,
        help="Directory containing trait network artefacts",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HOTSPOT_DIR,
        help="Directory to store hotspot ranking outputs",
    )

    # ---- --notes-dir ----
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=NOTES_DIR,
        help="Directory for Markdown hotspot briefs",
    )

    # ---- --overlay-path ----
    parser.add_argument(
        "--overlay-path",
        type=Path,
        help="Optional concordance overlay CSV to augment hotspot scoring",
    )

    # ---- --top-k ----
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override number of hotspot rows to retain",
    )

    # ---- --report-top-n ----
    parser.add_argument(
        "--report-top-n",
        type=int,
        help="Override number of Markdown briefs to emit",
    )

    # ---- --min-degree-percentile ----
    parser.add_argument(
        "--min-degree-percentile",
        type=float,
        help="Override degree percentile threshold for hub filtering",
    )

    # ---- --min-cross-domain-fraction ----
    parser.add_argument(
        "--min-cross-domain-fraction",
        type=float,
        help="Override minimum cross domain fraction for hubs",
    )

    # ---- --min-domain-diversity ----
    parser.add_argument(
        "--min-domain-diversity",
        type=int,
        help="Override minimum distinct neighbour domains",
    )

    # ---- --min-strength ----
    parser.add_argument(
        "--min-strength",
        type=float,
        help="Override minimum weighted degree (strength) threshold",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config


def load_network_assets(
    network_dir: Path,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Load graph, node summary, and cluster summary files."""

    graph_path = network_dir / "trait_network.graphml"
    node_path = network_dir / "trait_network_nodes.csv"
    cluster_path = network_dir / "trait_network_clusters.csv"

    missing: List[str] = [
        str(path)
        for path in [graph_path, node_path, cluster_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing network artefacts: " + ", ".join(missing)
        )

    graph = nx.read_graphml(graph_path)
    node_df = pd.read_csv(node_path)
    cluster_df = pd.read_csv(cluster_path)
    return graph, node_df, cluster_df


def load_overlay_stats(
    overlay_path: Optional[Path],
) -> Dict[str, Dict[str, float]]:
    """Compute concordance statistics per trait from overlay data."""

    if overlay_path is None:
        return {}
    if not overlay_path.exists():
        logger.warning("Overlay file not found at %s", overlay_path)
        return {}

    overlay_df = pd.read_csv(overlay_path)
    pair_df = overlay_df
    if not {
        "trait_a",
        "trait_b",
        "pair_direction_agreement",
    }.issubset(pair_df.columns):
        pair_path = overlay_path.with_name(
            f"{overlay_path.stem}_pairs{overlay_path.suffix}"
        )
        if pair_path.exists():
            pair_df = pd.read_csv(pair_path)
        else:
            logger.warning(
                "Overlay file lacks trait pair columns and pair file missing;"
                " concordance stats skipped",
            )
            return {}

    if pair_df.empty:
        logger.warning("Trait concordance overlay is empty")
        return {}

    direction_column = (
        "pair_direction_agreement"
        if "pair_direction_agreement" in pair_df.columns
        else "direction_concordance"
    )
    if direction_column not in pair_df.columns:
        logger.warning(
            "Overlay data missing direction column; concordance stats skipped"
        )
        return {}

    trait_records: List[pd.DataFrame] = []
    for trait_col in ("trait_a", "trait_b"):
        if trait_col not in pair_df.columns:
            continue
        subset = pair_df[[trait_col, direction_column]].copy()
        subset.rename(
            columns={
                trait_col: "trait_label",
                direction_column: "direction_score",
            },
            inplace=True,
        )
        if "pair_match_type" in pair_df.columns:
            subset["pair_match_type"] = pair_df["pair_match_type"]
        trait_records.append(subset)

    if not trait_records:
        logger.warning(
            "Overlay data did not include trait columns for concordance stats"
        )
        return {}

    combined = pd.concat(trait_records, ignore_index=True)
    combined.dropna(subset=["trait_label"], inplace=True)
    if combined.empty:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for trait, group in combined.groupby("trait_label"):
        direction_series = pd.to_numeric(
            group["direction_score"],
            errors="coerce",
        ).dropna()
        if direction_series.empty:
            continue
        mean_val = float(direction_series.mean())
        std_val = (
            float(direction_series.std(ddof=0))
            if len(direction_series) > 1
            else 0.0
        )
        count_val = float(len(direction_series))
        positive = float((direction_series > 0).sum())
        negative = float((direction_series < 0).sum())
        record: Dict[str, float] = {
            "concordance_mean": mean_val,
            "concordance_std": std_val,
            "concordance_count": count_val,
            "direction_positive": positive,
            "direction_negative": negative,
        }
        if "pair_match_type" in group.columns:
            match_counts = (
                group["pair_match_type"].dropna().value_counts().to_dict()
            )
            for match_type, value in match_counts.items():
                record[f"match_type_{match_type}_count"] = float(value)
        stats[str(trait)] = record

    return stats


def shannon_entropy(domain_counts: Counter) -> float:
    """Compute Shannon entropy for domain counts."""

    total = sum(domain_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in domain_counts.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy


def compute_z_scores(series: pd.Series) -> pd.Series:
    """Compute z-scores safely for a numeric series."""

    mean_val = series.mean()
    std_val = series.std(ddof=0)
    if std_val == 0 or math.isnan(std_val):
        return pd.Series(0.0, index=series.index)
    return (series - mean_val) / std_val


def build_hotspot_table(
    graph: nx.Graph,
    node_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    concordance_stats: Dict[str, Dict[str, float]],
    hub_config: Dict[str, Any],
    hotspot_config: Dict[str, Any],
) -> pd.DataFrame:
    """Assemble hub ranking metrics for all traits."""

    node_lookup = cast(
        Dict[str, Dict[str, Any]],
        node_df.set_index("trait_label").to_dict(orient="index"),
    )
    cluster_lookup = cast(
        Dict[int, Dict[str, Any]],
        cluster_df.set_index("cluster_id").to_dict(orient="index"),
    )

    def metric_float(
        container: Dict[str, Any], key: str, default: float = 0.0
    ) -> float:
        value = container.get(key, default)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(result):
            return default
        return result

    records: List[Dict[str, Any]] = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        neighbor_domains = [
            graph.nodes[neighbor].get("domain_label") for neighbor in neighbors
        ]
        domain_counts = Counter([label for label in neighbor_domains if label])

        metrics_row = node_lookup.get(node, {})

        cluster_raw = metrics_row.get("louvain_cluster", -1)
        try:
            cluster_id = int(cluster_raw)
        except (TypeError, ValueError):
            cluster_id = -1

        record = {
            "trait_label": node,
            "degree": float(graph.degree(node)),
            "strength": float(
                sum(
                    attr.get("weight", 0.0)
                    for _, _, attr in graph.edges(node, data=True)
                )
            ),
            "betweenness": metric_float(metrics_row, "betweenness"),
            "eigenvector": metric_float(metrics_row, "eigenvector"),
            "domain_label": graph.nodes[node].get("domain_label"),
            "domain_similarity": graph.nodes[node].get("domain_similarity"),
            "cluster_id": cluster_id,
            "neighbor_count": len(neighbors),
            "neighbor_domain_count": len(domain_counts),
            "neighbor_diversity": shannon_entropy(domain_counts),
            "cross_domain_fraction": (
                len(domain_counts) / len(neighbors)
                if len(neighbors) > 0
                else 0.0
            ),
        }

        cluster_row = cluster_lookup.get(cluster_id, {})
        record["cluster_density"] = metric_float(cluster_row, "density")
        record["cluster_cross_domain_fraction"] = metric_float(
            cluster_row,
            "cross_domain_fraction",
        )

        concordance = concordance_stats.get(node, {})
        record["concordance_mean"] = concordance.get("concordance_mean")
        record["concordance_std"] = concordance.get("concordance_std")
        record["concordance_count"] = concordance.get("concordance_count")
        record["concordance_positive"] = concordance.get("direction_positive")
        record["concordance_negative"] = concordance.get("direction_negative")
        for key, value in concordance.items():
            if key.startswith("match_type_"):
                record[key] = value

        records.append(record)

    hotspot_df = pd.DataFrame(records).set_index("trait_label")

    hotspot_df["degree_z"] = compute_z_scores(
        hotspot_df["degree"].astype(float)
    )
    hotspot_df["strength_z"] = compute_z_scores(
        hotspot_df["strength"].astype(float)
    )
    hotspot_df["betweenness_z"] = compute_z_scores(
        hotspot_df["betweenness"].astype(float)
    )
    hotspot_df["eigenvector_z"] = compute_z_scores(
        hotspot_df["eigenvector"].astype(float)
    )

    hotspot_df["hub_score"] = (
        hotspot_df["degree_z"]
        + hotspot_df["strength_z"]
        + 0.5 * hotspot_df["betweenness_z"]
        + 0.5 * hotspot_df["eigenvector_z"]
    )

    min_degree_percentile = hub_config.get("min_degree_percentile", 90)
    degree_threshold = hotspot_df["degree"].quantile(
        min_degree_percentile / 100
    )
    min_domain_diversity = hub_config.get("min_domain_diversity", 2)
    min_cross_fraction = hub_config.get("min_cross_domain_fraction", 0.25)
    min_strength = hub_config.get("min_strength", 0.0)

    filters = (
        (hotspot_df["degree"] >= degree_threshold)
        & (hotspot_df["neighbor_domain_count"] >= min_domain_diversity)
        & (hotspot_df["cross_domain_fraction"] >= min_cross_fraction)
        & (hotspot_df["strength"] >= min_strength)
    )

    concordance_cap = hotspot_config.get("max_concordance_std")
    if concordance_cap is not None:
        filters &= hotspot_df["concordance_std"].isna() | (
            hotspot_df["concordance_std"] <= concordance_cap
        )

    filtered = hotspot_df.loc[filters].copy()
    filtered.sort_values(by="hub_score", ascending=False, inplace=True)
    return filtered.reset_index()


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug from a trait label."""

    value_ascii = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    return value_ascii.strip("-") or "trait"


def write_hotspot_brief(
    trait_record: pd.Series,
    graph: nx.Graph,
    notes_dir: Path,
    top_neighbor_count: int = 5,
) -> Path:
    """Write Markdown summary for a hotspot trait."""

    trait_value = trait_record.get("trait_label", "")
    trait = str(trait_value)
    neighbors = graph[trait]
    top_neighbors = sorted(
        neighbors.items(),
        key=lambda pair: pair[1].get("weight", 0.0),
        reverse=True,
    )[:top_neighbor_count]

    lines: List[str] = []
    lines.append(f"Trait: {trait}")
    lines.append(f"Cluster id: {int(trait_record.get('cluster_id', -1))}")
    lines.append("Hub score: " + f"{trait_record.get('hub_score', 0.0):.3f}")
    lines.append("Degree: " + f"{trait_record.get('degree', 0.0):.0f}")
    lines.append("Strength: " + f"{trait_record.get('strength', 0.0):.1f}")
    lines.append(
        "Neighbor domain count: "
        + f"{trait_record.get('neighbor_domain_count', 0)}"
    )
    lines.append(
        "Neighbor diversity (Shannon): "
        + f"{trait_record.get('neighbor_diversity', 0.0):.3f}"
    )
    lines.append(
        "Cross domain fraction: "
        + f"{trait_record.get('cross_domain_fraction', 0.0):.2f}"
    )

    if pd.notna(trait_record.get("concordance_mean")):
        lines.append(
            "Concordance mean: "
            + f"{trait_record.get('concordance_mean', 0.0):.3f}"
        )
    if pd.notna(trait_record.get("concordance_std")):
        lines.append(
            "Concordance std: "
            + f"{trait_record.get('concordance_std', 0.0):.3f}"
        )
    if pd.notna(trait_record.get("concordance_count")):
        lines.append(
            "Concordance observations: "
            + f"{trait_record.get('concordance_count', 0.0):.0f}"
        )
    positive = trait_record.get("concordance_positive")
    negative = trait_record.get("concordance_negative")
    if pd.notna(positive) or pd.notna(negative):
        pos_display = f"{float(positive):.0f}" if pd.notna(positive) else "0"
        neg_display = f"{float(negative):.0f}" if pd.notna(negative) else "0"
        lines.append(
            "Direction breakdown: +" + pos_display + " / -" + neg_display
        )

    lines.append("")
    lines.append("Top neighbors:")
    for neighbor, data in top_neighbors:
        weight = data.get("weight", 0.0)
        lines.append(f"- {neighbor} ({weight:.0f} shared studies)")

    notes_dir.mkdir(parents=True, exist_ok=True)
    note_path = notes_dir / f"hotspot-{slugify(trait)}.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def export_hotspot_outputs(
    hotspot_df: pd.DataFrame,
    top_k: int,
    report_top_n: int,
    output_dir: Path,
    notes_dir: Path,
    graph: nx.Graph,
) -> Dict[str, Any]:
    """Export hotspot rankings and Markdown briefs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    top_df = hotspot_df.head(top_k).copy()
    csv_path = output_dir / "hotspot_rankings.csv"
    top_df.to_csv(csv_path, index=False)

    briefs: List[str] = []
    for _, row in top_df.head(report_top_n).iterrows():
        note_path = write_hotspot_brief(row, graph, notes_dir)
        briefs.append(str(note_path))

    metadata = {
        "ranking_csv": str(csv_path),
        "markdown_briefs": briefs,
        "record_count": int(len(top_df)),
    }
    metadata_path = output_dir / "hotspot_summary.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main() -> int:
    """Execute hotspot ranking workflow."""

    args = parse_args()
    config = load_config(args.config)

    hub_config = dict(config["case_study_2"]["hubs"])
    hotspot_config = dict(config["case_study_2"]["hotspots"])

    if args.min_degree_percentile is not None:
        hub_config["min_degree_percentile"] = args.min_degree_percentile
    if args.min_cross_domain_fraction is not None:
        hub_config["min_cross_domain_fraction"] = (
            args.min_cross_domain_fraction
        )
    if args.min_domain_diversity is not None:
        hub_config["min_domain_diversity"] = args.min_domain_diversity
    if args.min_strength is not None:
        hub_config["min_strength"] = args.min_strength
    if args.top_k is not None:
        hub_config["top_k"] = args.top_k
    if args.report_top_n is not None:
        hotspot_config["report_top_n"] = args.report_top_n

    logger.info(
        "Hotspot parameters: hubs=%s hotspots=%s", hub_config, hotspot_config
    )

    if args.dry_run:
        logger.info("Dry run complete")
        res = 0
        return res

    graph, node_df, cluster_df = load_network_assets(args.network_dir)
    concordance_stats = load_overlay_stats(args.overlay_path)

    hotspot_df = build_hotspot_table(
        graph,
        node_df,
        cluster_df,
        concordance_stats,
        hub_config,
        hotspot_config,
    )

    if hotspot_df.empty:
        logger.warning("No traits satisfied hotspot criteria")
        res = 0
        return res

    top_k = hub_config.get("top_k", 50)
    report_top_n = hotspot_config.get("report_top_n", min(10, top_k))

    metadata = export_hotspot_outputs(
        hotspot_df,
        top_k,
        report_top_n,
        args.output_dir,
        args.notes_dir,
        graph,
    )

    logger.info("Hotspot analysis complete: %s", metadata)
    res = 0
    return res


if __name__ == "__main__":
    raise SystemExit(main())
