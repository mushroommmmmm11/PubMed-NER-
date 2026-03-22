#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
急进高寒地区官兵主分析脚本（Python 版）。

功能目标：
1. 读取双表头 Excel 数据；
2. 构建主网络图、中心性图、桥接中心性图；
3. 导出边权矩阵、边表；
4. 进行 bootstrap 边权区间 / 差异分析；
5. 进行 case-dropping 稳定性分析；
6. 进行二分类组间网络比较与置换检验；
7. 进行探索性 DAG 搜索。

说明：
- 为保证运行更流畅，脚本将重复逻辑统一抽象为通用函数。
- 与原 R 版本相比，默认 bootstrap 次数更保守，可通过命令行调高。
- 某些分析（如 DAG）依赖可选三方包；缺失时会自动跳过并记录说明。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMPORT_ERRORS: List[str] = []

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    IMPORT_ERRORS.append("matplotlib")

try:
    import networkx as nx
except Exception:
    nx = None
    IMPORT_ERRORS.append("networkx")

try:
    import numpy as np
except Exception:
    np = None
    IMPORT_ERRORS.append("numpy")

try:
    import pandas as pd
except Exception:
    pd = None
    IMPORT_ERRORS.append("pandas")

try:
    from sklearn.covariance import GraphicalLasso
    from sklearn.preprocessing import StandardScaler
except Exception:
    GraphicalLasso = None
    StandardScaler = None
    IMPORT_ERRORS.append("scikit-learn")

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None
    IMPORT_ERRORS.append("scipy")

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional styling only
    sns = None

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional acceleration only
    Parallel = None
    delayed = None

try:
    from pgmpy.estimators import BICGauss, HillClimbSearch
except Exception:  # pragma: no cover - optional DAG analysis
    BICGauss = None
    HillClimbSearch = None

SEED = 20260313
RNG = np.random.default_rng(SEED) if np is not None else None

MAIN_NODES_FULL = [
    "说谎", "诈病", "性度", "轻躁狂", "社会内向", "疑病", "抑郁",
    "癔病Conversion Disorder", "精神病态", "妄想", "精神衰弱", "精神分裂症",
    "自然环境", "社会环境", "生理性改变",
    "信念及同伴自我", "社会自我", "负面自我", "军旅自我", "能力与品质自我", "成就自我",
    "整体功能",
]

SHORT_LABELS = [
    "L", "F", "Mf", "Ma", "Si", "Hs", "D", "Hy", "Pd", "Pa", "Pt", "Sc",
    "NE", "SE", "PC", "BP", "SS", "NS", "MS", "AT", "AS", "GAF",
]

COMMUNITIES = [
    *("MMPI" for _ in range(12)),
    *("SA" for _ in range(3)),
    *("SC" for _ in range(6)),
    "GAF",
]

GROUP_COLORS = {
    "MMPI": "#E64B35",
    "SA": "#4DBBD5",
    "SC": "#00A087",
    "GAF": "#3C5488",
}


@dataclass(frozen=True)
class AnalysisConfig:
    input_file: Path
    output_dir: Path
    group_var: str
    alpha: float
    bootstrap_iterations: int
    case_bootstrap_iterations: int
    nct_iterations: int
    dag_bootstrap_iterations: int
    glasso_alpha_grid: Tuple[float, ...]
    min_edge_abs: float
    max_workers: int


@dataclass
class BootstrapSummary:
    mean: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame


@dataclass
class NetworkArtifacts:
    corr: pd.DataFrame
    precision: pd.DataFrame
    partial_corr: pd.DataFrame
    graph: nx.Graph
    communities: Dict[str, str]
    layout: Dict[str, np.ndarray]


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="主分析网络脚本（Python 版）")
    parser.add_argument("--input-file", required=True, help="Excel 数据文件路径")
    parser.add_argument("--output-dir", default="main_analysis_all_figures_py", help="输出目录")
    parser.add_argument("--group-var", default="是否为独生子女", help="组间比较的二分类变量")
    parser.add_argument("--alpha", type=float, default=0.5, help="Graphical Lasso 收缩强度")
    parser.add_argument("--bootstrap-iterations", type=int, default=300, help="非参数 bootstrap 次数")
    parser.add_argument("--case-bootstrap-iterations", type=int, default=300, help="case-dropping 次数")
    parser.add_argument("--nct-iterations", type=int, default=300, help="组间置换检验次数")
    parser.add_argument("--dag-bootstrap-iterations", type=int, default=200, help="DAG bootstrap 次数")
    parser.add_argument("--min-edge-abs", type=float, default=0.02, help="绘图时保留的最小边权绝对值")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="并行 worker 数")
    args = parser.parse_args()

    return AnalysisConfig(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        group_var=args.group_var,
        alpha=args.alpha,
        bootstrap_iterations=max(50, args.bootstrap_iterations),
        case_bootstrap_iterations=max(50, args.case_bootstrap_iterations),
        nct_iterations=max(50, args.nct_iterations),
        dag_bootstrap_iterations=max(50, args.dag_bootstrap_iterations),
        glasso_alpha_grid=(args.alpha,),
        min_edge_abs=max(0.0, args.min_edge_abs),
        max_workers=max(1, args.max_workers),
    )


def clean_header(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.split("\n", 1)[0].strip()
    return text or "unnamed"


def make_unique_names(names: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    output: List[str] = []
    for name in names:
        base = name
        idx = counts.get(base, 0)
        if idx == 0:
            output.append(base)
        else:
            output.append(f"{base}_{idx}")
        counts[base] = idx + 1
    return output


def read_two_header_xlsx(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    header = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=2)
    names = make_unique_names([clean_header(value) for value in header.iloc[1].tolist()])
    data = pd.read_excel(path, sheet_name=sheet_name, skiprows=2, header=None)
    data.columns = names
    return data


def build_node_info() -> pd.DataFrame:
    node_info = pd.DataFrame(
        {
            "full": MAIN_NODES_FULL,
            "short": SHORT_LABELS,
            "community": COMMUNITIES,
        }
    )
    node_info["color"] = node_info["community"].map(GROUP_COLORS)
    return node_info


def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: pd.to_numeric(col, errors="coerce"))


def prepare_main_dataframe(raw_df: pd.DataFrame, node_info: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in MAIN_NODES_FULL if col not in raw_df.columns]
    if missing:
        raise KeyError(f"缺少主分析字段: {missing}")

    main_df = coerce_numeric_frame(raw_df[MAIN_NODES_FULL].copy())
    main_df.columns = node_info["short"].tolist()

    if "GAF" in main_df.columns:
        gaf_sorted = sorted(value for value in main_df["GAF"].dropna().unique())
        main_df["GAF"] = pd.Categorical(main_df["GAF"], categories=gaf_sorted, ordered=True)

    return main_df.dropna(axis=0, how="any").reset_index(drop=True)


def numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        if pd.api.types.is_categorical_dtype(converted[column]):
            converted[column] = converted[column].cat.codes.replace(-1, np.nan)
    return converted.astype(float)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = numeric_view(df)
    corr = numeric_df.corr(method="spearman")
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def fit_graphical_lasso(corr_df: pd.DataFrame, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr = corr_df.to_numpy(dtype=float)
    model = GraphicalLasso(alpha=alpha, max_iter=200, assume_centered=True)
    pseudo_samples = RNG.multivariate_normal(np.zeros(corr.shape[0]), corr, size=max(1000, corr.shape[0] * 60))
    model.fit(pseudo_samples)
    precision = pd.DataFrame(model.precision_, index=corr_df.index, columns=corr_df.columns)

    partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(partial, 0.0)
    partial_df = pd.DataFrame(partial, index=corr_df.index, columns=corr_df.columns)
    return precision, partial_df


def build_graph_from_weights(partial_corr: pd.DataFrame, communities: Dict[str, str], min_edge_abs: float) -> nx.Graph:
    graph = nx.Graph()
    for node in partial_corr.index:
        graph.add_node(node, community=communities[node], color=GROUP_COLORS[communities[node]])

    for i, left in enumerate(partial_corr.index):
        for j, right in enumerate(partial_corr.columns):
            if i >= j:
                continue
            weight = float(partial_corr.iat[i, j])
            if abs(weight) < min_edge_abs:
                continue
            graph.add_edge(left, right, weight=weight, sign="positive" if weight >= 0 else "negative")
    return graph


def estimate_network(df: pd.DataFrame, node_info: pd.DataFrame, alpha: float, min_edge_abs: float) -> NetworkArtifacts:
    communities = dict(zip(node_info["short"], node_info["community"]))
    corr = correlation_matrix(df)
    precision, partial_corr = fit_graphical_lasso(corr, alpha=alpha)
    graph = build_graph_from_weights(partial_corr, communities, min_edge_abs=min_edge_abs)
    layout = nx.spring_layout(graph, seed=SEED, weight="weight")
    return NetworkArtifacts(corr=corr, precision=precision, partial_corr=partial_corr, graph=graph, communities=communities, layout=layout)


def extract_edge_table(weights: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for i, left in enumerate(weights.index):
        for j in range(i + 1, len(weights.columns)):
            right = weights.columns[j]
            weight = float(weights.iat[i, j])
            if abs(weight) <= threshold:
                continue
            records.append({"from": left, "to": right, "weight": weight, "abs_weight": abs(weight)})
    edge_df = pd.DataFrame(records)
    if edge_df.empty:
        return edge_df
    return edge_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def plot_network(graph: nx.Graph, layout: Dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.axis("off")

    colors = [graph.nodes[node]["color"] for node in graph.nodes]
    widths = [1.0 + 4.0 * abs(graph.edges[edge]["weight"]) for edge in graph.edges]
    edge_colors = ["#D55E00" if graph.edges[edge]["weight"] > 0 else "#3B82F6" for edge in graph.edges]

    nx.draw_networkx_nodes(graph, layout, node_color=colors, node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, layout, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, layout, width=widths, edge_color=edge_colors, alpha=0.8, ax=ax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markersize=10)
        for name, color in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_centrality(graph: nx.Graph) -> pd.DataFrame:
    strength = {node: sum(abs(data["weight"]) for _, _, data in graph.edges(node, data=True)) for node in graph.nodes}
    expected_influence = {node: sum(data["weight"] for _, _, data in graph.edges(node, data=True)) for node in graph.nodes}

    distance_graph = nx.Graph()
    distance_graph.add_nodes_from(graph.nodes(data=True))
    for left, right, data in graph.edges(data=True):
        distance_graph.add_edge(left, right, distance=1.0 / (abs(data["weight"]) + 1e-9))

    if graph.number_of_edges() == 0:
        closeness = {node: 0.0 for node in graph.nodes}
        betweenness = {node: 0.0 for node in graph.nodes}
    else:
        closeness = nx.closeness_centrality(distance_graph, distance="distance")
        betweenness = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)

    centrality_df = pd.DataFrame(
        {
            "node": list(graph.nodes),
            "strength": pd.Series(strength),
            "closeness": pd.Series(closeness),
            "betweenness": pd.Series(betweenness),
            "expected_influence": pd.Series(expected_influence),
        }
    )
    return centrality_df.sort_values("expected_influence", ascending=False).reset_index(drop=True)


def zscore_series(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def plot_metric_table(metrics: pd.DataFrame, columns: Sequence[str], title: str, path: Path) -> None:
    plot_df = metrics.set_index("node")[list(columns)].apply(zscore_series)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    if sns is not None:
        sns.heatmap(plot_df, cmap="coolwarm", center=0, ax=ax)
    else:
        im = ax.imshow(plot_df.to_numpy(), cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(np.arange(plot_df.shape[1]), labels=plot_df.columns)
        ax.set_yticks(np.arange(plot_df.shape[0]), labels=plot_df.index)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_bridge_centrality(graph: nx.Graph, communities: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    nodes = list(graph.nodes)

    for node in nodes:
        bridge_strength = 0.0
        bridge_ei = 0.0
        cross_neighbors = set()
        for neighbor, data in graph[node].items():
            if communities[neighbor] == communities[node]:
                continue
            bridge_strength += abs(data["weight"])
            bridge_ei += data["weight"]
            cross_neighbors.add(neighbor)

        two_step = bridge_ei
        for neighbor in cross_neighbors:
            for second, data in graph[neighbor].items():
                if second == node or communities[second] == communities[node]:
                    continue
                two_step += data["weight"]

        rows.append(
            {
                "node": node,
                "community": communities[node],
                "bridge_strength": bridge_strength,
                "bridge_betweenness": 0.0,
                "bridge_closeness": len(cross_neighbors),
                "bridge_ei1": bridge_ei,
                "bridge_ei2": two_step,
            }
        )

    return pd.DataFrame(rows).sort_values("bridge_strength", ascending=False).reset_index(drop=True)


def resample_rows(df: pd.DataFrame, rng: np.random.Generator, fraction: float = 1.0) -> pd.DataFrame:
    n_samples = max(2, int(len(df) * fraction))
    indices = rng.integers(0, len(df), size=n_samples)
    return df.iloc[indices].reset_index(drop=True)


def single_bootstrap(df: pd.DataFrame, node_info: pd.DataFrame, alpha: float, min_edge_abs: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = resample_rows(df, rng=rng, fraction=1.0)
    return estimate_network(sampled, node_info, alpha=alpha, min_edge_abs=min_edge_abs).partial_corr


def run_parallel_map(func, items: Sequence[Any], max_workers: int) -> List[Any]:
    if Parallel is None or delayed is None or max_workers <= 1:
        return [func(item) for item in items]
    return Parallel(n_jobs=max_workers)(delayed(func)(item) for item in items)


def bootstrap_edges(df: pd.DataFrame, node_info: pd.DataFrame, config: AnalysisConfig) -> BootstrapSummary:
    seeds = [SEED + i for i in range(config.bootstrap_iterations)]

    def worker(seed: int) -> pd.DataFrame:
        return single_bootstrap(df, node_info, config.alpha, config.min_edge_abs, seed)

    mats = run_parallel_map(worker, seeds, max_workers=config.max_workers)
    stack = np.stack([mat.to_numpy() for mat in mats], axis=0)
    mean = pd.DataFrame(stack.mean(axis=0), index=mats[0].index, columns=mats[0].columns)
    lower = pd.DataFrame(np.quantile(stack, 0.025, axis=0), index=mats[0].index, columns=mats[0].columns)
    upper = pd.DataFrame(np.quantile(stack, 0.975, axis=0), index=mats[0].index, columns=mats[0].columns)
    return BootstrapSummary(mean=mean, lower=lower, upper=upper)


def bootstrap_centrality(df: pd.DataFrame, node_info: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for i in range(config.bootstrap_iterations):
        sampled = resample_rows(df, RNG, fraction=1.0)
        artifacts = estimate_network(sampled, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
        metrics = compute_centrality(artifacts.graph).set_index("node")
        metrics["iteration"] = i + 1
        rows.append(metrics.reset_index())
    return pd.concat(rows, ignore_index=True)


def summarise_metric_difference(boot_metrics: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    pivot = boot_metrics.pivot_table(index="iteration", columns="node", values=metric_name)
    records: List[Dict[str, Any]] = []
    columns = list(pivot.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            diff = pivot[left] - pivot[right]
            records.append(
                {
                    "node_a": left,
                    "node_b": right,
                    "mean_diff": float(diff.mean()),
                    "ci_lower": float(diff.quantile(0.025)),
                    "ci_upper": float(diff.quantile(0.975)),
                    "significant": bool(diff.quantile(0.025) > 0 or diff.quantile(0.975) < 0),
                }
            )
    return pd.DataFrame(records).sort_values(["significant", "mean_diff"], ascending=[False, False])


def plot_bootstrap_intervals(bootstrap: BootstrapSummary, path: Path) -> None:
    edge_df = extract_edge_table(bootstrap.mean)
    edge_df = edge_df.head(30).copy()
    if edge_df.empty:
        return
    lowers = []
    uppers = []
    labels = []
    means = []
    for _, row in edge_df.iterrows():
        labels.append(f"{row['from']}—{row['to']}")
        means.append(row["weight"])
        lowers.append(bootstrap.lower.loc[row["from"], row["to"]])
        uppers.append(bootstrap.upper.loc[row["from"], row["to"]])

    fig, ax = plt.subplots(figsize=(11, 8.5))
    y = np.arange(len(labels))
    ax.errorbar(
        x=means,
        y=y,
        xerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels=labels)
    ax.set_title("Bootstrap edge confidence intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_difference_heatmap(diff_df: pd.DataFrame, title: str, path: Path) -> None:
    if diff_df.empty:
        return
    top_df = diff_df.head(40).copy()
    heat = top_df[["mean_diff", "ci_lower", "ci_upper"]]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    if sns is not None:
        sns.heatmap(heat, cmap="coolwarm", center=0, annot=False, ax=ax, yticklabels=top_df["node_a"] + " vs " + top_df["node_b"])
    else:
        im = ax.imshow(heat.to_numpy(), cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_yticks(np.arange(len(top_df)), labels=(top_df["node_a"] + " vs " + top_df["node_b"]).tolist())
        ax.set_xticks(np.arange(heat.shape[1]), labels=heat.columns)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def case_drop_stability(df: pd.DataFrame, node_info: pd.DataFrame, config: AnalysisConfig, fractions: Sequence[float] = (0.9, 0.8, 0.7, 0.6, 0.5)) -> pd.DataFrame:
    full_graph = estimate_network(df, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs).graph
    full_metrics = compute_centrality(full_graph).set_index("node")

    records: List[Dict[str, Any]] = []
    for fraction in fractions:
        corrs_strength: List[float] = []
        corrs_ei: List[float] = []
        corrs_close: List[float] = []
        corrs_between: List[float] = []

        for _ in range(config.case_bootstrap_iterations):
            sampled = resample_rows(df, RNG, fraction=fraction)
            graph = estimate_network(sampled, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs).graph
            metrics = compute_centrality(graph).set_index("node")
            corrs_strength.append(spearmanr(full_metrics["strength"], metrics["strength"]).statistic)
            corrs_ei.append(spearmanr(full_metrics["expected_influence"], metrics["expected_influence"]).statistic)
            corrs_close.append(spearmanr(full_metrics["closeness"], metrics["closeness"]).statistic)
            corrs_between.append(spearmanr(full_metrics["betweenness"], metrics["betweenness"]).statistic)

        records.extend(
            [
                {"metric": "strength", "fraction_kept": fraction, "median_correlation": np.nanmedian(corrs_strength)},
                {"metric": "expected_influence", "fraction_kept": fraction, "median_correlation": np.nanmedian(corrs_ei)},
                {"metric": "closeness", "fraction_kept": fraction, "median_correlation": np.nanmedian(corrs_close)},
                {"metric": "betweenness", "fraction_kept": fraction, "median_correlation": np.nanmedian(corrs_between)},
            ]
        )

    return pd.DataFrame(records)


def compute_cs_coefficients(stability_df: pd.DataFrame, cutoff: float = 0.7) -> pd.DataFrame:
    rows = []
    for metric, sub_df in stability_df.groupby("metric"):
        passed = sub_df.loc[sub_df["median_correlation"] >= cutoff, "fraction_kept"]
        cs_value = 1.0 - float(passed.min()) if not passed.empty else 0.0
        rows.append({"metric": metric, "CS_coefficient": cs_value})
    return pd.DataFrame(rows)


def plot_case_drop_stability(stability_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for metric, sub_df in stability_df.groupby("metric"):
        ax.plot(sub_df["fraction_kept"], sub_df["median_correlation"], marker="o", label=metric)
    ax.axhline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Fraction of cases kept")
    ax.set_ylabel("Median Spearman correlation")
    ax.set_title("Case-dropping stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def split_groups(raw_df: pd.DataFrame, analysis_df: pd.DataFrame, group_var: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
    if group_var not in raw_df.columns:
        return None
    group_values = pd.to_numeric(raw_df[group_var], errors="coerce")
    main_numeric = numeric_view(analysis_df)
    valid = group_values.notna() & main_numeric.notna().all(axis=1)
    cmp_df = main_numeric.loc[valid].reset_index(drop=True)
    groups = group_values.loc[valid].astype(int).reset_index(drop=True)
    levels = sorted(groups.unique())
    if len(levels) != 2:
        return None
    group_a, group_b = levels
    df_a = cmp_df.loc[groups == group_a].reset_index(drop=True)
    df_b = cmp_df.loc[groups == group_b].reset_index(drop=True)
    return df_a, df_b, f"{group_var}_{group_a}", f"{group_var}_{group_b}"


def global_strength(weights: pd.DataFrame) -> float:
    upper = np.triu(np.abs(weights.to_numpy()), k=1)
    return float(upper.sum())


def nct_permutation(df_a: pd.DataFrame, df_b: pd.DataFrame, node_info: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    art_a = estimate_network(df_a, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
    art_b = estimate_network(df_b, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
    observed = global_strength(art_a.partial_corr) - global_strength(art_b.partial_corr)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    n_a = len(df_a)
    permuted_stats = []
    for _ in range(config.nct_iterations):
        perm = combined.sample(frac=1.0, replace=False, random_state=int(RNG.integers(0, 1_000_000)))
        perm_a = perm.iloc[:n_a].reset_index(drop=True)
        perm_b = perm.iloc[n_a:].reset_index(drop=True)
        stat = global_strength(
            estimate_network(perm_a, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs).partial_corr
        ) - global_strength(
            estimate_network(perm_b, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs).partial_corr
        )
        permuted_stats.append(stat)

    permuted = np.asarray(permuted_stats, dtype=float)
    p_value = float((np.abs(permuted) >= abs(observed)).mean())
    return pd.DataFrame(
        {
            "observed_global_strength_diff": [observed],
            "permutation_p_value": [p_value],
            "iterations": [config.nct_iterations],
            "group_a_n": [len(df_a)],
            "group_b_n": [len(df_b)],
        }
    )


def plot_group_networks(art_a: NetworkArtifacts, art_b: NetworkArtifacts, name_a: str, name_b: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, artifacts, title in zip(axes, [art_a, art_b], [name_a, name_b]):
        ax.set_title(title)
        ax.axis("off")
        graph = artifacts.graph
        nx.draw_networkx_nodes(graph, artifacts.layout, node_color=[graph.nodes[n]["color"] for n in graph.nodes], node_size=800, ax=ax)
        nx.draw_networkx_labels(graph, artifacts.layout, font_size=9, ax=ax)
        nx.draw_networkx_edges(graph, artifacts.layout, width=[1 + 4 * abs(graph.edges[e]["weight"]) for e in graph.edges], alpha=0.8, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_group_centrality(centrality_a: pd.DataFrame, centrality_b: pd.DataFrame, name_a: str, name_b: str, path: Path) -> None:
    merged = centrality_a[["node", "strength", "expected_influence"]].merge(
        centrality_b[["node", "strength", "expected_influence"]], on="node", suffixes=(f"_{name_a}", f"_{name_b}")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
    for ax, metric in zip(axes, ["strength", "expected_influence"]):
        width = np.arange(len(merged))
        ax.barh(width - 0.2, merged[f"{metric}_{name_a}"], height=0.4, label=name_a)
        ax.barh(width + 0.2, merged[f"{metric}_{name_b}"], height=0.4, label=name_b)
        ax.set_yticks(width, merged["node"])
        ax.set_title(metric)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_dag_analysis(df: pd.DataFrame, config: AnalysisConfig) -> Tuple[Optional[pd.DataFrame], Optional[nx.DiGraph], str]:
    if HillClimbSearch is None or BICGauss is None:
        return None, None, "未安装 pgmpy，已跳过 DAG 分析。"

    dag_df = numeric_view(df)
    dag_df = pd.DataFrame(StandardScaler().fit_transform(dag_df), columns=dag_df.columns)

    scorer = BICGauss(dag_df)
    estimator = HillClimbSearch(dag_df)
    model = estimator.estimate(scoring_method=scorer, max_indegree=4, show_progress=False)
    edges = list(model.edges())

    if not edges:
        return pd.DataFrame(columns=["from", "to", "boot_strength", "direction_prob"]), nx.DiGraph(), "DAG 未检出稳定弧。"

    counts: Dict[Tuple[str, str], int] = {}
    for i in range(config.dag_bootstrap_iterations):
        sample = dag_df.sample(frac=1.0, replace=True, random_state=SEED + i).reset_index(drop=True)
        scorer_i = BICGauss(sample)
        est_i = HillClimbSearch(sample)
        model_i = est_i.estimate(scoring_method=scorer_i, max_indegree=4, show_progress=False)
        for edge in model_i.edges():
            counts[edge] = counts.get(edge, 0) + 1

    rows = []
    for source, target in edges:
        strength = counts.get((source, target), 0) / config.dag_bootstrap_iterations
        reverse_strength = counts.get((target, source), 0) / config.dag_bootstrap_iterations
        rows.append(
            {
                "from": source,
                "to": target,
                "boot_strength": strength,
                "direction_prob": strength / max(strength + reverse_strength, 1e-9),
            }
        )

    dag_table = pd.DataFrame(rows).sort_values(["boot_strength", "direction_prob"], ascending=False).reset_index(drop=True)
    dag_graph = nx.DiGraph()
    dag_graph.add_nodes_from(df.columns)
    dag_graph.add_edges_from(edges)
    return dag_table, dag_graph, "DAG 分析完成。"


def plot_dag(graph: nx.DiGraph, node_info: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title("Exploratory DAG")
    ax.axis("off")
    colors = node_info.set_index("short")["color"].to_dict()
    layout = nx.spring_layout(graph, seed=SEED)
    nx.draw_networkx_nodes(graph, layout, node_size=850, node_color=[colors.get(node, "#999999") for node in graph.nodes], ax=ax)
    nx.draw_networkx_labels(graph, layout, font_size=9, ax=ax)
    nx.draw_networkx_edges(graph, layout, arrows=True, arrowstyle="-|>", arrowsize=15, width=1.5, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_session_info(config: AnalysisConfig) -> Dict[str, Any]:
    return {
        "python_version": os.sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "networkx_version": nx.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
        "config": {
            "input_file": str(config.input_file),
            "output_dir": str(config.output_dir),
            "group_var": config.group_var,
            "alpha": config.alpha,
            "bootstrap_iterations": config.bootstrap_iterations,
            "case_bootstrap_iterations": config.case_bootstrap_iterations,
            "nct_iterations": config.nct_iterations,
            "dag_bootstrap_iterations": config.dag_bootstrap_iterations,
        },
    }


def require_dependencies() -> None:
    if IMPORT_ERRORS:
        missing = ", ".join(sorted(set(IMPORT_ERRORS)))
        raise RuntimeError(f"缺少依赖，请先安装: {missing}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    config = parse_args()
    require_dependencies()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    node_info = build_node_info()
    raw_df = read_two_header_xlsx(config.input_file)
    numeric_df = coerce_numeric_frame(raw_df)
    main_df = prepare_main_dataframe(raw_df, node_info)

    node_info.to_csv(config.output_dir / "00_node_dictionary.csv", index=False, encoding="utf-8-sig")

    artifacts = estimate_network(main_df, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
    artifacts.partial_corr.round(3).to_csv(config.output_dir / "01_partial_correlation_matrix.csv", encoding="utf-8-sig")
    extract_edge_table(artifacts.partial_corr).to_csv(config.output_dir / "01_edge_table.csv", index=False, encoding="utf-8-sig")
    plot_network(artifacts.graph, artifacts.layout, "Main network: MMPI + SA + SC + GAF", config.output_dir / "02_main_network.png")

    centrality_df = compute_centrality(artifacts.graph)
    centrality_df.to_csv(config.output_dir / "03_centrality_table.csv", index=False, encoding="utf-8-sig")
    plot_metric_table(
        centrality_df,
        ["strength", "closeness", "betweenness", "expected_influence"],
        "Centrality z-scores",
        config.output_dir / "03_centrality_plot.png",
    )
    centrality_df[["node", "expected_influence"]].rename(columns={"expected_influence": "EI_1step"}).assign(EI_2step=lambda df: df["EI_1step"]).to_csv(
        config.output_dir / "03_expected_influence_table.csv", index=False, encoding="utf-8-sig"
    )

    bridge_df = compute_bridge_centrality(artifacts.graph, artifacts.communities)
    bridge_df.to_csv(config.output_dir / "04_bridge_centrality_table.csv", index=False, encoding="utf-8-sig")
    plot_metric_table(
        bridge_df.rename(columns={"bridge_strength": "strength", "bridge_closeness": "closeness", "bridge_betweenness": "betweenness", "bridge_ei1": "expected_influence"}),
        ["strength", "closeness", "betweenness", "expected_influence"],
        "Bridge centrality z-scores",
        config.output_dir / "04_bridge_centrality_plot.png",
    )

    edge_bootstrap = bootstrap_edges(main_df, node_info, config)
    edge_bootstrap.mean.to_csv(config.output_dir / "05_bootstrap_edge_mean.csv", encoding="utf-8-sig")
    edge_bootstrap.lower.to_csv(config.output_dir / "05_bootstrap_edge_ci_lower.csv", encoding="utf-8-sig")
    edge_bootstrap.upper.to_csv(config.output_dir / "05_bootstrap_edge_ci_upper.csv", encoding="utf-8-sig")
    plot_bootstrap_intervals(edge_bootstrap, config.output_dir / "05_edge_CI_bootstrap.png")

    boot_metrics = bootstrap_centrality(main_df, node_info, config)
    edge_diff = summarise_metric_difference(boot_metrics.rename(columns={"strength": "strength", "expected_influence": "expected_influence"}), "strength")
    edge_diff.to_csv(config.output_dir / "06_edge_difference_test.csv", index=False, encoding="utf-8-sig")
    plot_difference_heatmap(edge_diff, "Bootstrap strength difference", config.output_dir / "06_edge_difference_test.png")

    strength_diff = summarise_metric_difference(boot_metrics, "strength")
    strength_diff.to_csv(config.output_dir / "07_strength_difference_test.csv", index=False, encoding="utf-8-sig")
    plot_difference_heatmap(strength_diff, "Strength difference test", config.output_dir / "07_strength_difference_test.png")

    ei_diff = summarise_metric_difference(boot_metrics, "expected_influence")
    ei_diff.to_csv(config.output_dir / "08_expected_influence_difference_test.csv", index=False, encoding="utf-8-sig")
    plot_difference_heatmap(ei_diff, "Expected influence difference test", config.output_dir / "08_expected_influence_difference_test.png")

    stability_df = case_drop_stability(main_df, node_info, config)
    stability_df.to_csv(config.output_dir / "09_case_drop_stability.csv", index=False, encoding="utf-8-sig")
    plot_case_drop_stability(stability_df, config.output_dir / "09_case_drop_stability_plot.png")
    compute_cs_coefficients(stability_df).to_csv(config.output_dir / "09_CS_coefficients.csv", index=False, encoding="utf-8-sig")

    split = split_groups(numeric_df, main_df, config.group_var)
    if split is not None:
        group_a_df, group_b_df, name_a, name_b = split
        art_a = estimate_network(group_a_df, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
        art_b = estimate_network(group_b_df, node_info, alpha=config.alpha, min_edge_abs=config.min_edge_abs)
        art_a.partial_corr.to_csv(config.output_dir / "10_group1_network_matrix.csv", encoding="utf-8-sig")
        art_b.partial_corr.to_csv(config.output_dir / "10_group2_network_matrix.csv", encoding="utf-8-sig")
        art_b.layout = art_a.layout
        plot_group_networks(art_a, art_b, name_a, name_b, config.output_dir / "10_group_networks_side_by_side.png")
        plot_group_centrality(compute_centrality(art_a.graph), compute_centrality(art_b.graph), name_a, name_b, config.output_dir / "11_group_centrality_comparison.png")
        nct_df = nct_permutation(group_a_df, group_b_df, node_info, config)
        nct_df.to_csv(config.output_dir / "11_NCT_summary.csv", index=False, encoding="utf-8-sig")
    else:
        save_text(config.output_dir / "11_NCT_summary.csv", "group comparison skipped\n")

    dag_table, dag_graph, dag_message = run_dag_analysis(main_df, config)
    if dag_table is not None:
        dag_table.to_csv(config.output_dir / "12_DAG_arc_table.csv", index=False, encoding="utf-8-sig")
    if dag_graph is not None:
        plot_dag(dag_graph, node_info, config.output_dir / "12_exploratory_DAG.png")
    save_text(config.output_dir / "12_DAG_notes.txt", dag_message + "\n")

    save_json(config.output_dir / "99_sessionInfo.json", export_session_info(config))
    print(f"全部主分析图和表已导出到目录: {config.output_dir}")


if __name__ == "__main__":
    main()
