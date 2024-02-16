"""In this experiment, we will train a PICNN using moscot_not repository."""

from moscot.problems.generic._generic import ConditionalNeuralProblem
import jax.numpy as jnp
import scanpy as sc
import pickle as pkl
import jax
import os
from pathlib import Path
import sys
import wandb
import pandas as pd

# Internal imports
sys.path.append(
    '/Users/gori/Desktop/thesis/ConditionalOT_Perturbations/src/'
)
from evaluator import calculate_metrics

# Load the data
raw_data = sc.read_h5ad("/Users/gori/Desktop/thesis/ConditionalOT_Perturbations/Datasets/sciplex_complete_middle_subset_compare.h5ad")
raw_embedding = pkl.load(open("/Users/gori/Desktop/thesis/ConditionalOT_Perturbations/Datasets/processed_data_25.pickle", "rb"))

adata_train = raw_data[raw_data.obs["split_ood_finetuning"] == "train"]
adata_test = raw_data[raw_data.obs["split_ood_finetuning"] == "test"]
adata_ood = raw_data[raw_data.obs["split_ood_finetuning"] == "ood"]

inv_map = {
    "cell_lines": {v:k for k,v in raw_embedding["mapping"]["cell_lines"].items()},
    "conditions": {v:k for k,v in raw_embedding["mapping"]["conditions"].items()},
}

embedding_data = {}
embedding_data.update(
    {
       inv_map["cell_lines"][k]:v
        for k,v in raw_embedding["embedding_train"]["cell_lines"].items()
    }
)
embedding_data.update(
    {
        inv_map["conditions"][k]:v
        for k,v in raw_embedding["embedding_train"]["conditions"].items()
    }
)

neural_problem = ConditionalNeuralProblem(
    adata_train,
    embedding_data=embedding_data,
)

#Create training policy
subset = []
for cell_line in adata_train.obs["cell_type"].unique():
    for condition in adata_train[adata_train.obs["cell_type"] == cell_line].obs["cov_drug"].unique():
        subset.append(
            (f"{cell_line}_{'control'}", condition)
        )

neural_problem.prepare(
    key="cov_drug",
    joint_attr="X_pca",
    policy="explicit",
    subset=[subset[0]]
)

# prepare() method returns the pairs in inverse order, we want to solve from control to perturbation
neural_problem._sample_pairs = [(c, d) for (d,c) in neural_problem._sample_pairs]

neural_problem.solve(
    cond_dim=494,
    embedding_data=embedding_data,
    iterations=100,
    best_model_metric="sinkhorn"
)

batch_mapper = push_results = jax.vmap(
    lambda x, cond: neural_problem.solution.push(x=x, cond=cond),
)

"""Evaluation"""
with wandb.init(project="moscot_picnns", entity="gori") as run:
    for name, adata in zip(["ood", "test"], [adata_ood, adata_test]):
        print(f"INFO: Evaluating on {adata.obs['split_ood_finetuning'].unique()}")
        table = wandb.Table(
                    columns=["run", "cell_line", "condition"]
                )
        for cell_line_condition in adata.obs["cov_drug"].unique():
            cell_line, condition = cell_line_condition.split("_")
            if condition == "control":
                continue
            else:
                source_gex = adata[
                    (adata.obs["cell_type"] == cell_line) &
                    (adata.obs["condition"] == "control")
                ].obsm["X_pca"]

                try:
                    embedding = jnp.hstack([embedding_data[cell_line], embedding_data[condition]])
                except KeyError:
                    print(f"Skipping {cell_line}_{condition}")
                    continue

                target_gex = adata[
                    (adata.obs["cell_type"] == cell_line) &
                    (adata.obs["condition"] == condition)
                ].obsm["X_pca"]

                all_embeddings = jnp.repeat(
                    embedding[None, :],
                    target_gex.shape[0],
                    axis=0
                )

                batch_results = batch_mapper(
                    target_gex,
                    all_embeddings
                )

                metrics = calculate_metrics(
                    source=source_gex,
                    target=target_gex,
                    predicted=batch_results
                )
                table.add_data(
                    run.id,
                    inv_map["cell_lines"][cell_line],
                    inv_map["conditions"][condition],
                    *metrics.values(),
                )

            wandb.log({f"{name}_metrics": table})

            # Convert ood_metrics to a pandas DataFrame
            metrics_df = pd.DataFrame(
                table.data,
                columns=table.columns,
            )
            # Convert all columns except the first three to float
            metrics_df.iloc[:, 3:] = metrics_df.iloc[:, 3:].astype(float)

            # Log the aggregated metrics
            metrics_df_flat = metrics_df.drop(columns="condition")\
                                .groupby(["cell_line", "run"])\
                                .agg(['mean', 'std', 'min', 'max'])\
                                .reset_index()
            metrics_df_flat.columns = metrics_df_flat.columns.map('_'.join)
            metrics_df_flat.melt(
                id_vars=['cell_line_', "run_"],
                var_name='aggregated_metric',
                value_name='value'
            )

            # Create the table
            agg_metrics_table = wandb.Table(data=metrics_df_flat)

            # Log the table
            wandb.log({f"{name}_agg_metrics_table": agg_metrics_table})
        