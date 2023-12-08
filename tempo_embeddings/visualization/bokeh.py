import abc
import logging
from functools import lru_cache
from typing import Optional
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.palettes import turbo
from bokeh.plotting import Figure
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.transform import factor_cmap
from ..settings import OUTLIERS_LABEL
from ..text.abstractcorpus import AbstractCorpus
from .visualizer import Visualizer


class BokehVisualizer(Visualizer):
    """Base class for visualizers using the Bokeh library."""

    def __init__(self, *clusters: list[AbstractCorpus]):
        self._clusters = clusters

    def _select_palette(self):
        return turbo(len(self._clusters))

    @abc.abstractmethod
    def _create_data(self):
        return NotImplemented


class BokehInteractiveVisualizer(BokehVisualizer):
    """Interactive visualizer using the Bokeh library."""

    def _get_metadata_fields(self):
        return self._corpora[0].metadata_fields()

    @lru_cache
    def _create_data(self, label: Optional[str] = None) -> ColumnDataSource:
        return ColumnDataSource(
            pd.concat(
                (
                    pd.concat(
                        (
                            pd.DataFrame(cluster.hover_datas()),
                            cluster.embeddings_as_df(),
                        ),
                        axis="columns",
                    ).assign(label=cluster.label)
                    for cluster in self._clusters
                    if label is None or cluster.label == label
                )
            )
        )

    def _labels(self) -> list[str]:
        return np.unique(self._create_data().data["label"])

    def _create_clusters(
        self,
        columns,  # FIXME:set default value
        *,
        figure_height: int = 1500,  # TODO: scale size to number of clusters
        figure_width: int = 2000,
        legend_location: str = "right",
        click_policy: str = "hide",
    ):
        palette = self._select_palette()

        tool_tips = [
            ("corpus", "@label"),
            ("text", "@text"),
        ] + [(column, "@" + column) for column in columns]

        fig: Figure = figure(
            height=figure_height, width=figure_width, tooltips=tool_tips
        )

        for cluster in self._clusters:
            source = self._create_data(cluster.label)

            for _column in columns:
                if _column not in source.column_names:
                    logging.debug(
                        "Column '%s' not found in cluster '%s'.", _column, cluster.label
                    )

            glyph = fig.circle(
                source=source,
                x="x",
                y="y",
                color=factor_cmap("label", palette, self._labels()),
                legend_group="label",
            )
            if cluster.label == OUTLIERS_LABEL:
                glyph.visible = False

        legend = fig.legend[0]
        legend.label_text_font_size = "8px"
        legend.spacing = 0
        legend.location = legend_location
        legend.click_policy = click_policy

        return fig

    def visualize(self, *, metadata_fields=None):
        """Show an interactive plot.

        Args:
            metadata_fields: The metadata fields to include in the hover data.

        """

        scatter_plot = self._create_clusters(metadata_fields)

        show(scatter_plot)
