import abc
import logging
import pandas as pd
from bokeh.palettes import turbo
from bokeh.plotting import Figure
from bokeh.plotting import figure
from bokeh.plotting import show
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

    def _create_data(self):
        return pd.concat(
            (
                pd.concat(
                    (pd.DataFrame(cluster.hover_datas()), cluster.embeddings_as_df())
                ).assign(label=cluster.label)
                for cluster in self._clusters
            )
        )

    def _create_clusters(
        self,
        columns,  # FIXME:set default value
        *,
        figure_height: int = 1000,
        figure_width: int = 2000,
        legend_location: str = "right",
        click_policy: str = "hide",
    ):
        palette = self._select_palette()

        tool_tips = [(column, "@" + column) for column in columns] + [
            ("corpus", "@label"),
            ("text", "@text"),
        ]
        # tool_tips = [("year", "@year"), ("corpus", "@label")]

        fig: Figure = figure(
            height=figure_height, width=figure_width, tooltips=tool_tips
        )

        for i, cluster in enumerate(self._clusters):
            source = pd.concat(
                (
                    pd.DataFrame(cluster.hover_datas(columns)),
                    cluster.embeddings_as_df(),
                ),
                axis="columns",
            ).assign(label=cluster.label)

            glyph = fig.circle(
                source=source,
                x="x",
                y="y",
                color=palette[i],
                legend_label=cluster.label,
            )
            if cluster.label == OUTLIERS_LABEL:
                glyph.visible = False

            for column in columns:
                if column not in source.columns:
                    logging.warning(
                        "Column '%s' not found in cluster '%s'.", column, cluster.label
                    )

        legend = fig.legend[0]
        legend.location = legend_location
        legend.click_policy = click_policy

        return fig

    def visualize(self, *, metadata_fields=None):
        """Show an interactive plot.

        Args:
            metadata_fields: The metadata fields to include in the hover data.

        """

        show(self._create_clusters(metadata_fields))
