from unittest import mock

import pytest
from ipywidgets.widgets import (
    Button,
    HBox,
    Output,
    SelectionRangeSlider,
    SelectMultiple,
)

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.visualization.jscatter import JScatterVisualizer, PlotWidgets


@pytest.fixture
def mock_display():
    with mock.patch("tempo_embeddings.visualization.jscatter.display") as mock_display:
        yield mock_display


class TestJScatterVisualizer:
    @pytest.mark.parametrize(
        "cat_fields,cont_fields,cont_widget_types,cat_widget_types",
        [
            (["provenance"], ["year"], [], [SelectionRangeSlider, Output]),
            (["provenance"], [], [], []),
            ([], ["year"], [SelectMultiple, Output], [SelectionRangeSlider, Output]),
            (
                ["provenance", "invalid_1"],
                ["year", "invalid_2"],
                [],
                [SelectionRangeSlider, Output],
            ),
        ],
        # TODO: test for continuous filter
    )
    def test_visualize(
        self,
        mock_display,
        corpus,
        cat_fields,
        cont_fields,
        cont_widget_types,
        cat_widget_types,
    ):
        visualizer = JScatterVisualizer(
            corpus, categorical_fields=cat_fields, continuous_filter_fields=cont_fields
        )

        visualizer.visualize()

        widgets = mock_display.call_args.args

        categorical_filters = widgets[1].children
        continous_filters = widgets[2].children

        assert [type(w) for w in widgets] == [HBox, HBox, HBox, Button]
        assert [type(w) for w in continous_filters] == cont_widget_types
        assert [type(w) for w in categorical_filters] == cat_widget_types

        assert visualizer._cluster_plot is None

    def test_cluster_button(self, mock_display, corpus):
        visualizer = JScatterVisualizer(corpus)
        visualizer.visualize()

        widgets = mock_display.call_args.args

        assert visualizer.clusters is None

        widgets[-1].click()
        assert isinstance(visualizer._cluster_plot, PlotWidgets)
        assert all(isinstance(c, Corpus) for c in visualizer.clusters)
