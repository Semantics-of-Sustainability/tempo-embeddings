import pandas as pd
import seaborn as sns
from ..settings import OUTLIERS_LABEL
from .bokeh import BokehVisualizer


class ClusterVisualizer(BokehVisualizer):
    """Visualizer for clusters using Bokeh."""

    def _create_data(self, point_size: int = 10):
        rows = []

        for cluster in self._clusters:
            for embedding in cluster.embeddings:
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
                centroid = cluster.centroid()
                rows.append(
                    {
                        "x": centroid[0],
                        "y": centroid[1],
                        "cluster": cluster.label,
                        "size": point_size * 2,
                    }
                )

        return pd.DataFrame(rows)

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

        # FIXME: remove `size` from legend
        return sns.scatterplot(
            data=self._create_data(point_size=point_size),
            x="x",
            y="y",
            hue="cluster",
            palette=palette,
            size="size",
        )
