from types import NoneType

import pytest
from ipywidgets.widgets import Button, HBox, VBox

from tempo_embeddings.visualization.jscatter import JScatterVisualizer, PlotWidgets


class TestJScatterVisualizer:
    @pytest.mark.parametrize(
        "categorical_fields,continuous_filter_fields,expected_widget_types",
        [
            (["provenance"], ["year"], [HBox, NoneType, VBox, Button]),
            (["provenance"], [], [HBox, NoneType, Button]),
            ([], ["year"], [HBox, VBox, Button]),
            (
                ["provenance", "invalid_1"],
                ["year", "invalid_2"],
                [HBox, NoneType, NoneType, VBox, NoneType, Button],
            ),
        ],
    )
    def test_visualize(
        self,
        corpus,
        categorical_fields,
        continuous_filter_fields,
        expected_widget_types,
    ):
        visualizer = JScatterVisualizer(
            corpus,
            categorical_fields=categorical_fields,
            continuous_filter_fields=continuous_filter_fields,
        )

        widgets = visualizer.visualize()

        assert [type(w) for w in widgets] == expected_widget_types

        assert visualizer._cluster_plot is None

    def test_cluster_button(self, corpus):
        visualizer = JScatterVisualizer(corpus)
        widgets = visualizer.visualize()

        assert visualizer.clusters is None

        widgets[-1].click()
        assert isinstance(visualizer._cluster_plot, PlotWidgets)
        assert visualizer.clusters is not None
