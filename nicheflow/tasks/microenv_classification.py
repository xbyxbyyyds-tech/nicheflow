import numpy as np
from collections import Counter


def build_microenv_labels(ds):
    """
    为每个微环境构建 dominant cell type 标签

    Returns
    -------
    microenv_features: list[np.ndarray]
        每个微环境的 feature (简单做 mean pooling)

    microenv_labels: np.ndarray
        每个微环境的 dominant label (int)
    """

    microenv_features = []
    microenv_labels = []

    for tp in ds.timepoints_ordered:

        centers = ds.subsampled_timepoint_idx[tp]
        neighbor_dict = ds.timepoint_neighboring_indices[tp]

        for center_idx in centers:

            # 邻居 index
            neighbors = neighbor_dict[center_idx]

            # 包含自己
            all_cells = [center_idx] + neighbors
            all_cells = list(set(all_cells))      # 去重
            
            # === feature 聚合 ===
            features = ds.X_pca[all_cells]
            pooled_feature = features.mean(axis=0)

            # === 统计 dominant label ===
            cell_types = ds.ct[all_cells]
            counter = Counter(cell_types)
            dominant_ct = counter.most_common(1)[0][0]

            dominant_label = ds.ct_to_int[dominant_ct]

            microenv_features.append(pooled_feature)
            microenv_labels.append(dominant_label)

    microenv_features = np.stack(microenv_features)
    microenv_labels = np.array(microenv_labels)

    return microenv_features, microenv_labels