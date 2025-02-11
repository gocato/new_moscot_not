import itertools
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_numeric_dtype

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs_mixins import d_mixins
from moscot._types import ArrayLike, Numeric_t, Str_Dict_t
from moscot.base.output import BaseSolverOutput
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.birth_death import BirthDeathProblem
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K
from moscot.plotting._utils import set_plotting_vars
from moscot.utils.tagged_array import Tag

__all__ = ["TemporalMixin"]


# TODO(@MUCDK, @michalk8): check for ignore[misc] in line below, might become redundant
class TemporalMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):  # type:ignore
    """Protocol class."""

    adata: AnnData
    problems: Dict[Tuple[K, K], BirthDeathProblem]
    temporal_key: Optional[str]
    _temporal_key: Optional[str]

    def cell_transition(  # noqa: D102
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        ...

    def push(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:  # noqa: D102
        ...

    def pull(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        """Pull."""
        ...

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...

    def _sample_from_tmap(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Any], List[ArrayLike]]:
        ...

    def _compute_wasserstein_distance(
        self: "TemporalMixinProtocol[K, B]",
        point_cloud_1: ArrayLike,
        point_cloud_2: ArrayLike,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> Numeric_t:
        ...

    def _interpolate_gex_with_ot(
        self: "TemporalMixinProtocol[K, B]",
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        source: K,
        target: K,
        interpolation_parameter: float,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        ...

    def _get_data(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        intermediate: Optional[K] = None,
        target: Optional[K] = None,
        posterior_marginals: bool = True,
        *,
        only_start: bool = False,
    ) -> Union[Tuple[ArrayLike, AnnData], Tuple[ArrayLike, ArrayLike, ArrayLike, AnnData, ArrayLike]]:
        ...

    def _interpolate_gex_randomly(
        self: "TemporalMixinProtocol[K, B]",
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        interpolation_parameter: float,
        growth_rates: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        ...

    def _plot_temporal(
        self: "TemporalMixinProtocol[K, B]",
        data: Dict[K, ArrayLike],
        source: K,
        target: K,
        time_points: Optional[Iterable[K]] = None,
        basis: str = "umap",
        result_key: Optional[str] = None,
        fill_value: float = 0.0,
        save: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @staticmethod
    def _get_interp_param(
        source: K, intermediate: K, target: K, interpolation_parameter: Optional[float] = None
    ) -> Numeric_t:
        ...

    def __iter__(self) -> Iterator[Tuple[K, K]]:
        ...


class TemporalMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._temporal_key: Optional[str] = None

    @d_mixins.dedent
    def cell_transition(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> Optional[pd.DataFrame]:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        %(cell_trans_params)s
        %(forward_cell_transition)s
        %(aggregation_mode)s
        %(ott_jax_batch_size)s
        %(normalize)s
        %(key_added_plotting)s

        Returns
        -------
        %(return_cell_transition)s

        Notes
        -----
        %(notes_cell_transition)s
        """
        if TYPE_CHECKING:
            assert isinstance(self.temporal_key, str)
        return self._cell_transition(
            key=self.temporal_key,
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    @d_mixins.dedent
    def sankey(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        threshold: Optional[float] = None,
        normalize: bool = False,
        forward: bool = True,
        restrict_to_existing: bool = True,
        order_annotations: Optional[List[Any]] = None,
        key_added: Optional[str] = _constants.SANKEY,
        return_data: bool = False,
    ) -> Optional[List[pd.DataFrame]]:
        """
        Draw a Sankey diagram visualising transitions of cells across time points.

        Parameters
        ----------
        %(cell_trans_params)s
        %(threshold)s
        %(normalize)s
        %(forward)s
        %(restrict_to_existing)s
        %(order_annotations)s
        %(key_added_plotting)s
        %(return_data)s

        Returns
        -------
        Transition matrices of cells or groups of cells, as needed for a sankey.

        Notes
        -----
        To visualise the results, see :func:`moscot.plotting.sankey`.
        """
        tuples = self._policy.plan(start=source, end=target)
        cell_transitions = []
        for src, tgt in tuples:
            cell_transitions.append(
                self.cell_transition(
                    src,
                    tgt,
                    source_groups=source_groups,
                    target_groups=target_groups,
                    forward=forward,
                    normalize=normalize,
                )
            )

        if len(cell_transitions) > 1 and restrict_to_existing:
            for i in range(len(cell_transitions[:-1])):
                present_annotations = list(
                    set(cell_transitions[i + 1].index).intersection(set(cell_transitions[i].columns))
                )
                cell_transitions[i] = cell_transitions[i][present_annotations]

        if order_annotations is not None:
            cell_transitions_updated = []
            for ct in cell_transitions:
                order_annotations_present_index = [ann for ann in order_annotations if ann in ct.index]
                ct = ct.loc[order_annotations_present_index[::-1]]
                order_annotations_present_columns = [ann for ann in order_annotations if ann in ct.columns]
                ct = ct[order_annotations_present_columns[::-1]]
            cell_transitions_updated.append(ct)
        else:
            cell_transitions_updated = cell_transitions

        if threshold is not None:
            if threshold < 0:
                raise ValueError(f"Expected threshold to be non-negative, found `{threshold}`.")
            for ct in cell_transitions:
                ct[ct < threshold] = 0.0

        if isinstance(source_groups, str):
            key = source_groups
        # TODO(MUCDK): should this be really `target_groups`?
        elif isinstance(target_groups, dict):
            key = list(target_groups.keys())[0]
        else:
            raise TypeError(f"Expected early cells to be either `str` or `dict`, found `{type(source_groups)}`.")

        if key_added is not None:
            plot_vars = {
                "transition_matrices": cell_transitions_updated,
                "key": key,
                "source": source,
                "target": target,
                "source_groups": source_groups,
                "target_groups": target_groups,
                "captions": [str(t) for t in tuples],
            }
            set_plotting_vars(self.adata, _constants.SANKEY, key=key_added, value=plot_vars)
        return cell_transitions_updated if return_data else None

    @d_mixins.dedent
    def push(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PUSH,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Push distribution of cells through time.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=True,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )

        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "source": source,
                "target": target,
                "temporal_key": self.temporal_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)
            set_plotting_vars(self.adata, _constants.PUSH, key=key_added, value=plot_vars)
        return result if return_data else None

    @d_mixins.dedent
    def pull(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PULL,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Pull distribution of cells through time.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=False,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )
        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "temporal_key": self.temporal_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
                "source": source,
                "target": target,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)
            set_plotting_vars(self.adata, _constants.PULL, key=key_added, value=plot_vars)
        return result if return_data else None

    @property
    def prior_growth_rates(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Return the prior estimate of growth rates of the cells in the source distribution."""
        # TODO(michalk8): FIXME
        cols = ["prior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.prior_growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup].adata_tgt.obs), 1),
                    fill_value=np.nan,
                ),
                index=self.problems[tup].adata_tgt.obs.index,
                columns=cols,
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    @property
    def posterior_growth_rates(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Return the posterior estimate of growth rates of the cells in the source distribution."""
        # TODO(michalk8): FIXME
        cols = ["posterior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.posterior_growth_rates, index=problem.adata.obs.index, columns=cols)
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(
                    shape=(len(self.problems[tup].adata_tgt.obs), 1),
                    fill_value=np.nan,
                ),
                index=self.problems[tup].adata_tgt.obs.index,
                columns=cols,
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    # TODO(michalk8): refactor me
    @property
    def cell_costs_source(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Return the cost of a cell obtained by the potentials of the optimal transport solution."""
        sol = list(self.problems.values())[0].solution
        if TYPE_CHECKING:
            assert isinstance(sol, BaseSolverOutput)
        if sol.potentials is None:
            return None
        df_list = [
            pd.DataFrame(
                np.array(np.abs(problem.solution.potentials[0])),  # type: ignore[union-attr,index]
                index=problem.adata_src.obs_names,
                columns=["cell_cost_source"],
            )
            for problem in self.problems.values()
        ]
        tup = list(self)[-1]
        df_list.append(
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup].adata_tgt.obs), 1), fill_value=np.nan),
                index=self.problems[tup].adata_tgt.obs_names,
                columns=["cell_cost_source"],
            )
        )
        return pd.concat(df_list, verify_integrity=True)

    @property
    def cell_costs_target(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Return the cost of a cell (see online methods) obtained by the potentials of the OT solution."""
        sol = list(self.problems.values())[0].solution
        if TYPE_CHECKING:
            assert isinstance(sol, BaseSolverOutput)
        if sol.potentials is None:
            return None

        tup = list(self)[0]
        df_list = [
            pd.DataFrame(
                np.full(shape=(len(self.problems[tup].adata_src), 1), fill_value=np.nan),
                index=self.problems[tup].adata_src.obs_names,
                columns=["cell_cost_target"],
            )
        ]
        df_list.extend(
            [
                pd.DataFrame(
                    np.array(np.abs(problem.solution.potentials[1])),  # type: ignore[union-attr,index]
                    index=problem.adata_tgt.obs_names,
                    columns=["cell_cost_target"],
                )
                for problem in self.problems.values()
            ]
        )
        return pd.concat(df_list, verify_integrity=True)

    # TODO(michalk8): refactor me
    def _get_data(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: Optional[K] = None,
        target: Optional[K] = None,
        posterior_marginals: bool = True,
        *,
        only_start: bool = False,
    ) -> Union[Tuple[ArrayLike, AnnData], Tuple[ArrayLike, ArrayLike, ArrayLike, AnnData, ArrayLike]]:
        # TODO: use .items()
        for src, tgt in self.problems:
            tag = self.problems[src, tgt].xy.tag  # type: ignore[union-attr]
            if tag != Tag.POINT_CLOUD:
                raise ValueError(
                    f"Expected `tag={Tag.POINT_CLOUD}`, "  # type: ignore[union-attr]
                    f"found `tag={self.problems[src, tgt].xy.tag}`."
                )
            if src == source:
                source_data = self.problems[src, tgt].xy.data_src  # type: ignore[union-attr]
                if only_start:
                    return source_data, self.problems[src, tgt].adata_src
                # TODO(michalk8): posterior marginals
                attr = "posterior_growth_rates" if posterior_marginals else "prior_growth_rates"
                growth_rates_source = getattr(self.problems[src, tgt], attr)
                break
        else:
            raise ValueError(f"No data found for `{source}` time point.")
        for src, tgt in self.problems:
            if src == intermediate:
                intermediate_data = self.problems[src, tgt].xy.data_src  # type: ignore[union-attr]
                intermediate_adata = self.problems[src, tgt].adata_src
                break
        else:
            raise ValueError(f"No data found for `{intermediate}` time point.")
        for src, tgt in self.problems:
            if tgt == target:
                target_data = self.problems[src, tgt].xy.data_tgt  # type: ignore[union-attr]
                break
        else:
            raise ValueError(f"No data found for `{target}` time point.")

        return (  # type:ignore[return-value]
            source_data,
            growth_rates_source,
            intermediate_data,
            intermediate_adata,
            target_data,
        )

    @d_mixins.dedent
    def compute_interpolated_distance(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        interpolation_parameter: Optional[float] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        batch_size: int = 256,
        posterior_marginals: bool = True,
        seed: Optional[int] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> Numeric_t:
        """
        Compute the Wasserstein distance between the OT-interpolated distribution and the true cell distribution.

        This is a validation method which interpolates the cell distributions corresponding to `start` and `end`
        leveraging the OT coupling to obtain an approximation of the cell distribution at time point `intermediate`.
        Therefore, the Wasserstein distance between the interpolated and the real distribution is computed.

        It is recommended to compare the Wasserstein distance to the ones obtained by
        :meth:`compute_time_point_distances` and :meth:`compute_random_distance`.

        This method does not instantiate the transport matrix if the solver output does not.

        TODO: link to notebook


        Parameters
        ----------
        %(start)s
        %(intermediate_interpolation)s
        %(end)s
        %(interpolation_parameter)s
        %(n_interpolated_cells)s
        %(account_for_unbalancedness)s
        %(ott_jax_batch_size)s
        %(use_posterior_marginals)s
        %(seed_sampling)s
        %(backend)s
        %(kwargs_divergence)s

        Returns
        -------
        Wasserstein distance between OT-based interpolated distribution and the true cell distribution.
        """
        source_data, _, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source,
            intermediate,
            target,
            posterior_marginals=posterior_marginals,
            only_start=False,
        )
        interpolation_parameter = self._get_interp_param(
            source, intermediate, target, interpolation_parameter=interpolation_parameter
        )
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)
        interpolation = self._interpolate_gex_with_ot(
            number_cells=n_interpolated_cells,
            source_data=source_data,
            target_data=target_data,
            source=source,
            target=target,
            interpolation_parameter=interpolation_parameter,
            account_for_unbalancedness=account_for_unbalancedness,
            batch_size=batch_size,
            seed=seed,
        )
        return self._compute_wasserstein_distance(
            point_cloud_1=intermediate_data, point_cloud_2=interpolation, backend=backend, **kwargs
        )

    @d_mixins.dedent
    def compute_random_distance(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        interpolation_parameter: Optional[float] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        posterior_marginals: bool = True,
        seed: Optional[int] = None,
        backend: Literal["ott"] = "ott",  # TODO: not used
        **kwargs: Any,
    ) -> Numeric_t:
        """
        Compute the Wasserstein distance of a randomly interpolated cell distribution and the true cell distribution.

        This method interpolates the cell trajectories at the `intermediate` time point using a random coupling and
        computes the distance to the true cell distribution.

        TODO: link to notebook

        Parameters
        ----------
        %(start)s
        %(intermediate_interpolation)s
        %(end)s
        %(interpolation_parameter)s
        %(n_interpolated_cells)s
        %(account_for_unbalancedness)s
        %(use_posterior_marginals)s
        %(seed_interpolation)s
        %(backend)s
        %(kwargs_divergence)s

        Returns
        -------
        The Wasserstein distance between a randomly interpolated cell distribution and the true cell distribution.
        """
        source_data, growth_rates_source, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source, intermediate, target, posterior_marginals=posterior_marginals, only_start=False
        )

        interpolation_parameter = self._get_interp_param(
            source, intermediate, target, interpolation_parameter=interpolation_parameter
        )
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)

        growth_rates = growth_rates_source if account_for_unbalancedness else None
        random_interpolation = self._interpolate_gex_randomly(
            n_interpolated_cells,
            source_data,
            target_data,
            interpolation_parameter=interpolation_parameter,
            growth_rates=growth_rates,
            seed=seed,
        )
        return self._compute_wasserstein_distance(intermediate_data, random_interpolation, **kwargs)

    @d_mixins.dedent
    def compute_time_point_distances(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        posterior_marginals: bool = True,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> Tuple[Numeric_t, Numeric_t]:
        """
        Compute the Wasserstein distance of cell distributions between time points.

        This method computes the Wasserstein distance between the cell distribution corresponding to `start` and `
        intermediate` and `intermediate` and `end`, respectively.

        TODO: link to notebook

        Parameters
        ----------
        %(start)s
        %(intermediate)s
        %(end)s
        %(use_posterior_marginals)s
        %(backend)s
        %(kwargs_divergence)s
        """
        source_data, _, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source,
            intermediate,
            target,
            posterior_marginals=posterior_marginals,
            only_start=False,
        )
        distance_source_intermediate = self._compute_wasserstein_distance(
            point_cloud_1=source_data, point_cloud_2=intermediate_data, backend=backend, **kwargs
        )
        distance_intermediate_target = self._compute_wasserstein_distance(
            point_cloud_1=intermediate_data, point_cloud_2=target_data, backend=backend, **kwargs
        )

        return distance_source_intermediate, distance_intermediate_target

    def compute_batch_distances(
        self: TemporalMixinProtocol[K, B],
        time: K,
        batch_key: str,
        posterior_marginals: bool = True,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> np.float_:
        """
        Compute the mean Wasserstein distance between batches of a distribution corresponding to one time point.

        Parameters
        ----------
        %(time_batch_distance)s
        %(batch_key_batch_distance)s
        %(use_posterior_marginals)s
        %(backend)s
        %(kwargs_divergence)

        Returns
        -------
        The mean Wasserstein distance between batches of a distribution corresponding to one time point.
        """
        data, adata = self._get_data(time, posterior_marginals=posterior_marginals, only_start=True)  # type: ignore[misc] # noqa: E501
        assert len(adata) == len(data), "TODO: wrong shapes"
        dist: List[Numeric_t] = []
        for batch_1, batch_2 in itertools.combinations(adata.obs[batch_key].unique(), 2):
            dist.append(
                self._compute_wasserstein_distance(
                    point_cloud_1=data[(adata.obs[batch_key] == batch_1).values, :],
                    point_cloud_2=data[(adata.obs[batch_key] == batch_2).values, :],
                    backend=backend,
                    **kwargs,
                )
            )
        return np.mean(dist)

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate,
    # faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self: TemporalMixinProtocol[K, B],
        point_cloud_1: ArrayLike,
        point_cloud_2: ArrayLike,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> Numeric_t:
        if backend == "ott":
            from moscot.backends.ott._utils import _compute_sinkhorn_divergence

            distance = _compute_sinkhorn_divergence(point_cloud_1, point_cloud_2, a, b, **kwargs)
        else:
            raise NotImplementedError("Only `ott` available as backend.")
        return distance

    def _interpolate_gex_with_ot(
        self: TemporalMixinProtocol[K, B],
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        source: K,
        target: K,
        interpolation_parameter: float,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        rows_sampled, cols_sampled = self._sample_from_tmap(
            source=source,
            target=target,
            n_samples=number_cells,
            source_dim=len(source_data),
            target_dim=len(target_data),
            batch_size=batch_size,
            account_for_unbalancedness=account_for_unbalancedness,
            interpolation_parameter=interpolation_parameter,
            seed=seed,
        )
        return (
            source_data[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
            + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
        )

    def _interpolate_gex_randomly(
        self: TemporalMixinProtocol[K, B],
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        interpolation_parameter: float,
        growth_rates: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        rng = np.random.RandomState(seed)
        if growth_rates is None:
            row_probability = np.ones(len(source_data))
        else:
            row_probability = growth_rates ** (1 - interpolation_parameter)
        row_probability /= np.sum(row_probability)
        return (
            source_data[rng.choice(len(source_data), size=number_cells, p=row_probability), :]
            * (1 - interpolation_parameter)
            + target_data[rng.choice(len(target_data), size=number_cells), :] * interpolation_parameter
        )

    @staticmethod
    def _get_interp_param(
        source: K, intermediate: K, target: K, interpolation_parameter: Optional[float] = None
    ) -> Numeric_t:
        if TYPE_CHECKING:
            assert isinstance(source, float)
            assert isinstance(intermediate, float)
            assert isinstance(target, float)
        if interpolation_parameter is not None:
            if 0 < interpolation_parameter < 1:
                return interpolation_parameter
            raise ValueError(
                f"Expected interpolation parameter to be in interval `(0, 1)`, found `{interpolation_parameter}`."
            )

        if source < intermediate < target:
            return (intermediate - source) / (target - source)
        raise ValueError(
            f"Expected intermediate time point to be in interval `({source}, {target})`, found `{intermediate}`."
        )

    @property
    def temporal_key(self) -> Optional[str]:
        """Temporal key in :attr:`anndata.AnnData.obs`."""
        return self._temporal_key

    @temporal_key.setter
    def temporal_key(self: TemporalMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find temporal key in `adata.obs[{key!r}]`.")
        if key is not None and not is_numeric_dtype(self.adata.obs[key]):
            raise TypeError(
                "Temporal key has to be of numeric type."
                f"Found `adata.obs[{key!r}]` to be of type `{infer_dtype(self.adata.obs[key])}`."
            )
        self._temporal_key = key
