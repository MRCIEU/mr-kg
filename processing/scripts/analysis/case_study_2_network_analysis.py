"""Construct trait co-occurrence networks and run community detection.

This script ingests the trait co-occurrence outputs from the profile
preparation step, builds filtered trait networks, annotates traits with EFO
metadata, performs community detection, and exports node/cluster summaries
along with graph serialisations for downstream hotspot and overlay analyses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from networkx.algorithms import community as nx_community
from yiutils.project_utils import find_project_root

try:  # python-louvain import name
    from community import community_louvain
except ImportError:  # pragma: no cover - handled in tests
    raise ImportError(
        "python-louvain package is required for community detection"
    )

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
COOCCURRENCE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "cooccurrence"
)
NETWORK_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "network"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for network construction."""

    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without writing outputs",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file path (default: {DEFAULT_CONFIG})",
    )

    # ---- --cooccurrence-dir ----
    parser.add_argument(
        "--cooccurrence-dir",
        type=Path,
        help="Directory containing co-occurrence outputs from preparation step",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store network artefacts (overrides configuration)",
    )

    # ---- --min-edge-weight ----
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        help="Override minimum edge weight (shared studies)",
    )

    # ---- --min-semantic-similarity ----
    parser.add_argument(
        "--min-semantic-similarity",
        type=float,
        help="Override minimum semantic similarity threshold",
    )

    # ---- --limit-traits ----
    parser.add_argument(
        "--limit-traits",
        type=int,
        help="Optional limit on number of traits processed (for sampling)",
    )

    res = parser.parse_args()
    return res


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure or NaN."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration for the case studies."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config


def resolve_paths(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Resolve filesystem paths and parameter overrides."""

    cs2_config = config["case_study_2"]
    network_config = cs2_config["network"]
    output_config = config["output"]["case_study_2"]

    cooccurrence_dir = args.cooccurrence_dir or (
        PROJECT_ROOT / output_config["cooccurrence"]
    )
    output_dir = args.output_dir or (PROJECT_ROOT / output_config["network"])

    parameters = {
        "edge_weight_threshold": args.min_edge_weight
        if args.min_edge_weight is not None
        else network_config["edge_weight_threshold"],
        "semantic_similarity_threshold": args.min_semantic_similarity
        if args.min_semantic_similarity is not None
        else network_config["semantic_similarity_threshold"],
        "jaccard_threshold": network_config.get("jaccard_threshold", 0.0),
        "min_degree": network_config.get("min_degree", 1),
        "max_neighbors": network_config.get("max_neighbors"),
        "louvain": network_config.get("louvain", {}),
        "label_propagation": network_config.get("label_propagation", {}),
        "centrality_metrics": network_config.get("centrality_metrics", []),
        "limit_traits": args.limit_traits,
    }

    return cooccurrence_dir, output_dir, parameters


def load_cooccurrence_tables(
    cooccurrence_dir: Path,
    limit_traits: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pair metrics and trait frequency tables from disk."""

    pair_path = cooccurrence_dir / "trait_pair_metrics.csv"
    trait_path = cooccurrence_dir / "trait_frequency.csv"

    if not pair_path.exists() or not trait_path.exists():
        raise FileNotFoundError(
            "Expected co-occurrence outputs not found. Run the prepare "
            "profiles script first."
        )

    pair_df = pd.read_csv(pair_path)
    trait_df = pd.read_csv(trait_path)

    if limit_traits is not None and limit_traits > 0:
        selected_list = trait_df.head(limit_traits)["trait_label"].tolist()
        mask = pair_df["trait_a"].isin(selected_list) & pair_df[
            "trait_b"
        ].isin(selected_list)
        pair_df = pair_df.loc[mask].reset_index(drop=True)
        trait_df = trait_df.loc[
            trait_df["trait_label"].isin(selected_list)
        ].reset_index(drop=True)
        logger.info(
            "Applied trait limit: %s traits, %s edges",
            len(trait_df),
            len(pair_df),
        )

    return pair_df, trait_df


def filter_pair_records(
    pair_df: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Filter edges according to configured thresholds."""

    mask = (
        (pair_df["shared_studies"] >= parameters["edge_weight_threshold"])
        & (
            pair_df["semantic_similarity"].fillna(1.0)
            >= parameters["semantic_similarity_threshold"]
        )
        & (pair_df["jaccard"] >= parameters["jaccard_threshold"])
    )

    filtered = pair_df.loc[mask].reset_index(drop=True)
    logger.info(
        "Filtered edges from %s to %s records",
        len(pair_df),
        len(filtered),
    )
    return filtered


def build_graph(
    pair_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    parameters: Dict[str, Any],
) -> nx.Graph:
    """Construct a weighted trait network from pair metrics."""

    graph = nx.Graph()

    for _, row in trait_df.iterrows():
        graph.add_node(
            row["trait_label"],
            study_count=int(row["study_count"]),
        )

    for _, row in pair_df.iterrows():
        trait_a = row["trait_a"]
        trait_b = row["trait_b"]
        shared_value = int(safe_float(row.get("shared_studies"), default=0.0))
        weight = float(shared_value)
        sem_raw = row.get("semantic_similarity")
        semantic_attr = None
        if not pd.isna(sem_raw):
            semantic_attr = safe_float(sem_raw, default=0.0)
        attributes = {
            "weight": weight,
            "shared_studies": shared_value,
            "jaccard": safe_float(row.get("jaccard"), default=0.0),
            "dice": safe_float(row.get("dice"), default=0.0),
            "lift": safe_float(row.get("lift"), default=0.0),
            "cooccurrence_rate": safe_float(
                row.get("cooccurrence_rate"),
                default=0.0,
            ),
            "semantic_similarity": semantic_attr,
        }
        graph.add_edge(trait_a, trait_b, **attributes)

    max_neighbors = parameters.get("max_neighbors")
    if max_neighbors:
        for node in list(graph.nodes()):
            neighbors = sorted(
                graph[node].items(),
                key=lambda item: item[1].get("weight", 0.0),
                reverse=True,
            )
            for neighbor, _ in neighbors[max_neighbors:]:
                if graph.has_edge(node, neighbor):
                    graph.remove_edge(node, neighbor)

    min_degree = parameters.get("min_degree", 1)
    if min_degree > 0:
        to_remove = [
            node for node, degree in graph.degree() if degree < min_degree
        ]
        graph.remove_nodes_from(to_remove)
        logger.info(
            "Removed %s nodes below degree %s", len(to_remove), min_degree
        )

    graph.remove_nodes_from(list(nx.isolates(graph)))
    logger.info(
        "Constructed graph with %s nodes and %s edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def attach_vector_store(
    paths_config: Dict[str, Any],
) -> duckdb.DuckDBPyConnection:
    """Attach the vector store database for trait annotations."""

    vector_db = PROJECT_ROOT / paths_config["vector_store"]
    if not vector_db.exists():
        raise FileNotFoundError(
            f"Vector store database not found: {vector_db}"
        )

    conn = duckdb.connect(database=":memory:")
    vector_path = str(vector_db.resolve()).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{vector_path}' AS vector_db (READ_ONLY)")
    return conn


def fetch_trait_domains(
    conn: duckdb.DuckDBPyConnection,
    trait_labels: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Fetch top EFO domain annotations for the provided traits."""

    labels = sorted({label for label in trait_labels if label})
    if not labels:
        return {}

    placeholders = ", ".join(["?"] * len(labels))
    query = f"""
        SELECT trait_label, efo_id, efo_label, similarity
        FROM (
            SELECT
                trait_label,
                efo_id,
                efo_label,
                similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY trait_label ORDER BY similarity DESC
                ) AS rn
            FROM vector_db.trait_efo_similarity_search
            WHERE trait_label IN ({placeholders})
        )
        WHERE rn = 1
    """

    rows = conn.execute(query, labels).fetchall()
    annotations: Dict[str, Dict[str, Any]] = {}
    for trait_label, efo_id, efo_label, similarity in rows:
        annotations[trait_label] = {
            "efo_id": efo_id,
            "efo_label": efo_label,
            "domain_similarity": float(similarity),
        }

    return annotations


def annotate_graph_nodes(
    graph: nx.Graph,
    annotations: Dict[str, Dict[str, Any]],
) -> None:
    """Annotate graph nodes with domain metadata where available."""

    for node in graph.nodes():
        annotation = annotations.get(node, {})
        graph.nodes[node]["efo_id"] = annotation.get("efo_id")
        graph.nodes[node]["domain_label"] = annotation.get("efo_label")
        graph.nodes[node]["domain_similarity"] = annotation.get(
            "domain_similarity"
        )


def run_community_detection(
    graph: nx.Graph,
    parameters: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
    """Run Louvain and label propagation clustering on the graph."""

    metadata: Dict[str, Any] = {}
    louvain_params = parameters.get("louvain", {})
    if graph.number_of_nodes() > 0:
        partition = community_louvain.best_partition(
            graph,
            weight="weight",
            resolution=louvain_params.get("resolution", 1.0),
            random_state=louvain_params.get("random_state"),
        )
        metadata["louvain_modularity"] = community_louvain.modularity(
            partition,
            graph,
            weight="weight",
        )
    else:
        partition = {}
        metadata["louvain_modularity"] = None

    label_assignments: Dict[str, int] = {}
    if graph.number_of_nodes() > 0:
        communities = list(
            nx_community.asyn_lpa_communities(
                graph,
                weight="weight",
                seed=louvain_params.get("random_state"),
            )
        )
        for cluster_id, community_nodes in enumerate(communities):
            for node in community_nodes:
                label_assignments[node] = cluster_id
        metadata["label_propagation_cluster_count"] = len(communities)
    else:
        metadata["label_propagation_cluster_count"] = 0

    metadata["louvain_cluster_count"] = len(set(partition.values()))
    return partition, label_assignments, metadata


def compute_centrality_metrics(
    graph: nx.Graph,
    requested_metrics: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute requested centrality metrics for graph nodes."""

    metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    if "degree" in requested_metrics:
        degree_scores = dict(graph.degree(weight=None))
        for node, value in degree_scores.items():
            metrics[node]["degree"] = float(value)

    if "strength" in requested_metrics:
        strength_scores = graph.degree(weight="weight")
        for node, value in strength_scores:
            metrics[node]["strength"] = float(value)

    if "betweenness" in requested_metrics:
        betweenness = nx.betweenness_centrality(graph, weight="weight")
        for node, value in betweenness.items():
            metrics[node]["betweenness"] = float(value)

    if "eigenvector" in requested_metrics and graph.number_of_nodes() > 0:
        try:
            eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
            for node, value in eigen.items():
                metrics[node]["eigenvector"] = float(value)
        except nx.NetworkXException as error:  # pragma: no cover - rare branch
            logger.warning("Eigenvector centrality failed: %s", error)

    return metrics


def build_node_dataframe(
    graph: nx.Graph,
    louvain_partition: Dict[str, int],
    label_partition: Dict[str, int],
    metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Assemble node-level DataFrame with annotations and metrics."""

    records: List[Dict[str, Any]] = []
    for node in graph.nodes():
        data = graph.nodes[node]
        metric_values = metrics.get(node, {})
        degree_value = metric_values.get("degree", float(graph.degree(node)))
        strength_value = metric_values.get(
            "strength",
            float(
                sum(
                    attr.get("weight", 0.0)
                    for _, _, attr in graph.edges(node, data=True)
                )
            ),
        )
        record = {
            "trait_label": node,
            "study_count": data.get("study_count"),
            "efo_id": data.get("efo_id"),
            "domain_label": data.get("domain_label"),
            "domain_similarity": data.get("domain_similarity"),
            "louvain_cluster": louvain_partition.get(node),
            "label_propagation_cluster": label_partition.get(node),
            "degree": float(degree_value),
            "strength": float(strength_value),
        }
        for key, value in metric_values.items():
            record[key] = float(value)
        records.append(record)

    node_df = pd.DataFrame(records)
    return node_df


def summarise_clusters(
    node_df: pd.DataFrame,
    graph: nx.Graph,
    partition: Dict[str, int],
) -> pd.DataFrame:
    """Compute cluster-level summaries for Louvain communities."""

    cluster_records: List[Dict[str, Any]] = []
    groups = defaultdict(list)
    for node, cluster_id in partition.items():
        groups[cluster_id].append(node)

    for cluster_id, members in groups.items():
        subgraph = graph.subgraph(members)
        domain_counts = Counter(
            [graph.nodes[node].get("domain_label") for node in members]
        )
        if None in domain_counts:
            del domain_counts[None]
        total_nodes = len(members)
        distinct_domains = len(domain_counts)
        cross_domain_fraction = (
            distinct_domains / total_nodes if total_nodes > 0 else 0.0
        )

        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "size": total_nodes,
                "density": nx.density(subgraph) if total_nodes > 1 else 0.0,
                "average_weight": np.mean(
                    [
                        data.get("weight", 0.0)
                        for _, _, data in subgraph.edges(data=True)
                    ]
                )
                if subgraph.number_of_edges() > 0
                else 0.0,
                "distinct_domains": distinct_domains,
                "cross_domain_fraction": cross_domain_fraction,
                "top_domains": ", ".join(
                    f"{label} ({count})"
                    for label, count in domain_counts.most_common(3)
                ),
            }
        )

    cluster_df = pd.DataFrame(cluster_records).sort_values(
        by=["size", "density"], ascending=[False, False]
    )
    return cluster_df.reset_index(drop=True)


def export_graph_files(
    graph: nx.Graph,
    node_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Export graph to JSON and GraphML representations."""

    graph_json = output_dir / "trait_network.json"
    graphml_path = output_dir / "trait_network.graphml"

    nodes_payload = cast(
        List[Dict[str, Any]],
        node_df.to_dict(orient="records"),
    )
    edges_payload: List[Dict[str, Any]] = []
    for source, target, data in graph.edges(data=True):
        edge_record = {
            "source": source,
            "target": target,
            **{key: value for key, value in data.items()},
        }
        edges_payload.append(edge_record)

    with graph_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {"nodes": nodes_payload, "edges": edges_payload}, handle, indent=2
        )

    nx.write_graphml(graph, graphml_path)
    return graph_json, graphml_path


def export_outputs(
    output_dir: Path,
    node_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    graph_json: Path,
    graphml_path: Path,
    metadata: Dict[str, Any],
    parameters: Dict[str, Any],
    graph_stats: Dict[str, Any],
) -> None:
    """Persist node, cluster, and metadata artefacts."""

    node_csv = output_dir / "trait_network_nodes.csv"
    cluster_csv = output_dir / "trait_network_clusters.csv"
    metadata_path = output_dir / "trait_network_metadata.json"

    node_df.to_csv(node_csv, index=False)
    cluster_df.to_csv(cluster_csv, index=False)

    payload = {
        "parameters": parameters,
        "graph": graph_stats,
        "files": {
            "node_csv": str(node_csv),
            "cluster_csv": str(cluster_csv),
            "graph_json": str(graph_json),
            "graphml": str(graphml_path),
        },
    }
    payload.update(metadata)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Exported node, cluster, and metadata artefacts")


def main() -> int:
    """Execute trait network construction and community detection."""

    args = parse_args()
    config = load_config(args.config)
    cooccurrence_dir, output_dir, parameters = resolve_paths(args, config)

    logger.info("Network construction parameters: %s", parameters)

    if args.dry_run:
        logger.info("Dry run successful")
        res = 0
        return res

    pair_df, trait_df = load_cooccurrence_tables(
        cooccurrence_dir,
        parameters["limit_traits"],
    )

    filtered_pairs = filter_pair_records(pair_df, parameters)
    if filtered_pairs.empty:
        logger.warning("No edges pass the configured thresholds")
        res = 0
        return res

    graph = build_graph(filtered_pairs, trait_df, parameters)
    if graph.number_of_edges() == 0:
        logger.warning("Graph has no edges after filtering")
        res = 0
        return res

    output_dir.mkdir(parents=True, exist_ok=True)

    conn = attach_vector_store(config["databases"])
    annotations = fetch_trait_domains(conn, graph.nodes())
    conn.close()
    annotate_graph_nodes(graph, annotations)

    louvain_partition, label_partition, community_metadata = (
        run_community_detection(
            graph,
            parameters,
        )
    )

    requested_metrics = set(parameters.get("centrality_metrics", []))
    requested_metrics.add("degree")
    requested_metrics.add("strength")
    metrics = compute_centrality_metrics(graph, sorted(requested_metrics))

    node_df = build_node_dataframe(
        graph, louvain_partition, label_partition, metrics
    )
    cluster_df = summarise_clusters(node_df, graph, louvain_partition)

    graph_json, graphml_path = export_graph_files(graph, node_df, output_dir)
    edges_payload = cast(
        List[Dict[str, Any]],
        filtered_pairs.to_dict(orient="records"),
    )
    metadata = {
        **community_metadata,
        "edges": edges_payload,
    }
    graph_stats = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
    }

    export_outputs(
        output_dir,
        node_df,
        cluster_df,
        graph_json,
        graphml_path,
        metadata,
        parameters,
        graph_stats,
    )

    logger.info("Network analysis completed")
    res = 0
    return res


if __name__ == "__main__":
    raise SystemExit(main())
