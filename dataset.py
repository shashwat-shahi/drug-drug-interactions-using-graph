"""Dataset utilities for DDI prediction with PyTorch."""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Generator, Dict, Tuple, Optional
from ogb.linkproppred import LinkPropPredDataset


@dataclass
class Pairs:
    """Represents positive and negative pairs of drug-drug interactions."""

    pos: np.ndarray
    neg: np.ndarray

    def get_eval_batches(
        self, batch_size: int
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Generates evaluation batches of positive and negative pairs."""
        n_pairs = self._n_pairs()
        indices = np.arange(n_pairs)

        for i in range(self._n_batches(batch_size)):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            yield {
                "pos": self.pos[batch_indices],
                "neg": self.neg[batch_indices]
            }

    def get_train_batches(
        self, batch_size: int, rng: np.random.Generator
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """Generates shuffled training batches with sampled negative pairs."""
        # Shuffle indices for positive pairs
        indices = np.arange(self._n_pairs())
        rng.shuffle(indices)

        # Get sample of negative pairs
        neg_sample = self._global_negative_sampling(rng)

        for i in range(self._n_batches(batch_size)):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            yield {
                "pos": self.pos[batch_indices],
                "neg": neg_sample[batch_indices]
            }

    def _global_negative_sampling(self, rng: np.random.Generator) -> np.ndarray:
        """Samples negative pairs from the entire set to match positive set size."""
        return rng.choice(self.neg, size=self.pos.shape[0], replace=False)

    def get_dummy_input(self) -> Dict[str, np.ndarray]:
        """Returns a small dummy subset of positive and negative pairs."""
        return {
            "pos": self.pos[:2],
            "neg": self.neg[:2]
        }

    def _n_batches(self, batch_size: int) -> int:
        """Calculates number of batches in the dataset given a batch size."""
        return int(np.floor(self._n_pairs() / batch_size))

    def _n_pairs(self) -> int:
        """Returns the smaller number of positive or negative pairs."""
        return int(min(self.pos.shape[0], self.neg.shape[0]))


@dataclass
class Dataset:
    """Graph dataset with nodes, pairs, and optional annotations."""

    n_nodes: int
    graph: Dict[str, np.ndarray]  # Contains 'senders', 'receivers', and 'edge_index'
    pairs: Pairs
    annotation: pd.DataFrame = field(default_factory=pd.DataFrame)

    @staticmethod
    def subset(
        dataset_dict: Dict[str, "Dataset"],
        node_ids: np.ndarray,
        keep_original_ids: bool = False,
    ) -> Dict[str, "Dataset"]:
        """Creates subset of dataset by keeping only specified nodes."""
        subsetted_datasets = {}

        # Create mapping from old node IDs to new ones
        if keep_original_ids:
            node_id_map = {nid: nid for nid in node_ids}
        else:
            node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}

        for name, dataset in dataset_dict.items():
            # Filter edges to only those between nodes in node_ids
            edge_mask = np.isin(dataset.graph["senders"], node_ids) & \
                        np.isin(dataset.graph["receivers"], node_ids)

            new_senders = np.array([node_id_map[s] for s in dataset.graph["senders"][edge_mask]])
            new_receivers = np.array([node_id_map[r] for r in dataset.graph["receivers"][edge_mask]])

            # Filter pairs
            pos_mask = np.isin(dataset.pairs.pos[:, 0], node_ids) & \
                       np.isin(dataset.pairs.pos[:, 1], node_ids)
            neg_mask = np.isin(dataset.pairs.neg[:, 0], node_ids) & \
                       np.isin(dataset.pairs.neg[:, 1], node_ids)

            new_pos = np.array([[node_id_map[p[0]], node_id_map[p[1]]]
                               for p in dataset.pairs.pos[pos_mask]])
            new_neg = np.array([[node_id_map[n[0]], node_id_map[n[1]]]
                               for n in dataset.pairs.neg[neg_mask]])

            # Filter annotation if exists
            if not dataset.annotation.empty:
                new_annotation = dataset.annotation[dataset.annotation['node_id'].isin(node_ids)].copy()
                if not keep_original_ids:
                    new_annotation['node_id'] = new_annotation['node_id'].map(node_id_map)
            else:
                new_annotation = pd.DataFrame()

            new_dataset = Dataset(
                n_nodes=len(node_ids),
                graph={
                    "senders": new_senders,
                    "receivers": new_receivers,
                    "edge_index": np.vstack([new_senders, new_receivers])
                },
                pairs=Pairs(pos=new_pos, neg=new_neg),
                annotation=new_annotation
            )
            subsetted_datasets[name] = new_dataset

        return subsetted_datasets


class DatasetBuilder:
    """Builds DDI datasets from the OGB benchmark."""

    def __init__(self, path: str):
        """Initializes the dataset builder with a path to the dataset."""
        self.path = path

    def build(
        self,
        node_limit: Optional[int] = None,
        rng_seed: int = 42,
        keep_original_ids: bool = False,
    ) -> Dict[str, Dataset]:
        """Builds and returns a dictionary of dataset splits."""
        dataset_splits = {}
        n_nodes, split_pairs = self._download()
        annotation = self._prepare_annotation()

        for name, split in split_pairs.items():
            pos_pairs = split["edge"]
            neg_pairs = split.get("edge_neg", None)
            graph = self._prepare_graph(n_nodes, pos_pairs)
            pairs = self._prepare_pairs(graph, pos_pairs, neg_pairs)
            dataset_splits[name] = Dataset(n_nodes, graph, pairs, annotation)

        if node_limit:
            rng = np.random.default_rng(rng_seed)
            node_ids = rng.choice(np.arange(n_nodes), size=node_limit, replace=False)
            dataset_splits = Dataset.subset(dataset_splits, node_ids, keep_original_ids)

        return dataset_splits

    def _download(self) -> Tuple[int, Dict]:
        """Downloads the dataset and returns the number of nodes and edge splits."""
        raw = LinkPropPredDataset(name="ogbl-ddi", root=self.path)
        n_nodes = raw[0]["num_nodes"]
        split_pairs = raw.get_edge_split()
        split_pairs["train"]["edge_neg"] = None  # Placeholder for negative edges
        return n_nodes, split_pairs

    def _prepare_annotation(self) -> pd.DataFrame:
        """Annotates nodes by mapping node IDs to database IDs and drug names."""
        try:
            ddi_descriptions = pd.read_csv(
                f"{self.path}/ogbl_ddi/mapping/ddi_description.csv.gz"
            )
            node_to_dbid_lookup = pd.read_csv(
                f"{self.path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
            )

            # Merge first and second drug descriptions
            first_drug = ddi_descriptions.loc[
                :, ["first drug id", "first drug name"]
            ].rename(columns={"first drug id": "dbid", "first drug name": "drug_name"})

            second_drug = ddi_descriptions.loc[
                :, ["second drug id", "second drug name"]
            ].rename(columns={"second drug id": "dbid", "second drug name": "drug_name"})

            dbid_to_name_lookup = (
                pd.concat([first_drug, second_drug])
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # Merge with node-to-DBID lookup
            annotation = pd.merge(
                node_to_dbid_lookup.rename(
                    columns={"drug id": "dbid", "node idx": "node_id"}
                ),
                dbid_to_name_lookup,
                on="dbid",
                how="inner",
            )
            return annotation
        except Exception as e:
            print(f"Warning: Could not load annotation: {e}")
            return pd.DataFrame()

    def _prepare_graph(
        self, n_nodes: int, pos_pairs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Prepares a graph from positive edge pairs."""
        senders, receivers = self._make_undirected(pos_pairs[:, 0], pos_pairs[:, 1])
        return {
            "senders": senders,
            "receivers": receivers,
            "edge_index": np.vstack([senders, receivers])
        }

    @staticmethod
    def _make_undirected(
        senders: np.ndarray, receivers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Makes an undirected graph by duplicating edges in both directions."""
        senders_undir = np.concatenate((senders, receivers))
        receivers_undir = np.concatenate((receivers, senders))
        return senders_undir, receivers_undir

    def _prepare_pairs(
        self,
        graph: Dict[str, np.ndarray],
        pos_pairs: np.ndarray,
        neg_pairs: Optional[np.ndarray] = None,
    ) -> Pairs:
        """Prepares positive and negative edge pairs."""
        if neg_pairs is None:
            neg_pairs = self._infer_negative_pairs(graph, pos_pairs.max() + 1)
        return Pairs(pos=pos_pairs, neg=neg_pairs)

    @staticmethod
    def _infer_negative_pairs(
        graph: Dict[str, np.ndarray], n_nodes: int
    ) -> np.ndarray:
        """Infers negative edge pairs in a graph."""
        # Initialize a matrix where all possible edges are marked as potential negative edges
        neg_adj_mask = np.ones((n_nodes, n_nodes), dtype=np.uint8)

        # Mask out existing edges in the graph
        neg_adj_mask[graph["senders"], graph["receivers"]] = 0

        # Use the upper triangular part to avoid duplicate pairs and self-loops
        neg_adj_mask = np.triu(neg_adj_mask, k=1)
        neg_pairs = np.array(neg_adj_mask.nonzero()).T
        return neg_pairs


def optimal_batch_size(
    dataset_splits: Dict[str, Dataset], remainder_tolerance: float = 0.125
) -> int:
    """Calculates optimal batch size for PyTorch DataLoader."""
    # Calculate the minimum length of positive and negative pairs for each dataset
    lengths = [
        min(dataset.pairs.pos.shape[0], dataset.pairs.neg.shape[0])
        for dataset in dataset_splits.values()
    ]

    # Determine the allowable remainders
    remainder_thresholds = [
        int(length * remainder_tolerance) for length in lengths
    ]
    max_possible_batch_size = min(lengths)

    for batch_size in range(max_possible_batch_size, 0, -1):
        remainders = [length % batch_size for length in lengths]
        if all(
            remainder <= threshold
            for remainder, threshold in zip(remainders, remainder_thresholds)
        ):
            return batch_size
    return max_possible_batch_size
