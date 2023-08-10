import pandas as pd
import seaborn as sns
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category10_10
from bokeh.palettes import Category20_20
from bokeh.plotting import Figure
from bokeh.plotting import figure
from ..settings import OUTLIERS_LABEL
from ..text.corpus import Corpus
from .visualizer import Visualizer


class ClusterVisualizer(Visualizer):
    _palettes = {10: Category10_10, 20: Category20_20}

    def __init__(self, clusters: list[Corpus]):
        self._clusters = clusters

    def _select_palette(self):
        try:
            return next(
                palette
                for size, palette in ClusterVisualizer._palettes.items()
                if size >= len(self._clusters)
            )
        except StopIteration as e:
            raise ValueError(
                f"Too many clusters ({len(self._clusters)}) for available palettes."
            ) from e

    def visualize(self, palette=None, point_size: int = 10):
        """Create a Scatter plot of the clusters.

        Args:
            palette: The palette to use for the clusters;
                defaults to a built-in palette suitable for the number of clusters
            point_size: The size of the points in the plot
        """
        palette = palette or self._select_palette()

        if len(self._clusters) > len(palette):
            raise ValueError(
                f"Too many clusters ({len(self._clusters)}) "
                f"for palette ({len(palette)})."
            )

        rows = []

        for cluster in self._clusters:
            print("-------------------------------")
            print(cluster.label)
            for passage, distance in cluster.nearest_neighbours():
                print(distance, passage)

            for embedding in cluster.umap_embeddings():
                rows.append(
                    {
                        "x": embedding[0],
                        "y": embedding[1],
                        "cluster": cluster.label,
                        "size": point_size,
                    }
                )

            if cluster.label != OUTLIERS_LABEL:
                # Add point for centroid
                centroid = cluster.umap_mean()
                rows.append(
                    {
                        "x": centroid[0],
                        "y": centroid[1],
                        "cluster": cluster.label,
                        "size": point_size * 2,
                    }
                )

        data = pd.DataFrame(rows)
        # FIXME: remove `size` from legend
        return sns.scatterplot(
            data=data, x="x", y="y", hue="cluster", palette=palette, size="size"
        )

    def interactive(self, palette=None, size: int = 10) -> Figure:
        """Generate an interactive plot.

        Args:
            palette: The palette to use for the clusters;
                defaults to a built-in palette suitable for the number of clusters
            point_size: The size of the points in the plot

        Returns: a Figure object for use with bokeh.plotting.show()
        """
        palette = palette or self._select_palette()

        if len(self._clusters) > len(palette):
            raise ValueError(
                f"Too many clusters ({len(self._clusters)}) "
                f"for palette ({len(palette)})."
            )

        p: Figure = figure()

        for i, cluster in enumerate(self._clusters):
            if cluster.label != OUTLIERS_LABEL:
                centroid = cluster.umap_mean()

                # FIXME: fill in or remove empty hover data
                p.circle(
                    x=centroid[0],
                    y=centroid[1],
                    size=size * 2,
                    color=palette[i],
                    fill_alpha=0.1,
                )

            _data = pd.DataFrame(cluster.hover_datas())
            embeddings = cluster.umap_embeddings()

            assert len(_data) == len(embeddings)

            _data["x"] = [e[0] for e in embeddings]
            _data["y"] = [e[1] for e in embeddings]

            tool_tips = [
                (column, "@" + column)
                for column in _data.columns
                if column not in ("x", "y")
            ]

            p.add_tools(HoverTool(tooltips=tool_tips))
            p.circle(source=_data, x="x", y="y", size=size, color=palette[i])

        return p
