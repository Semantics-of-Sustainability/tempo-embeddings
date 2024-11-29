from contextlib import nullcontext as does_not_raise
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
from tempo_embeddings.visualization.jscatter import JScatterVisualizer
from tempo_embeddings.visualization.util import DownloadButton


@pytest.fixture
def mock_display():
    with mock.patch("tempo_embeddings.visualization.jscatter.display") as mock_display:
        yield mock_display


class TestJScatterVisualizer:
    @pytest.mark.parametrize(
        "cat_fields,cont_fields,expected_cont_widget_types,expected_cat_widget_types,exception",
        [
            (["provenance"], ["year"], [], [SelectionRangeSlider, Output], None),
            (["provenance"], [], [], [], None),
            (
                [],
                ["year"],
                [SelectMultiple, Output, SelectMultiple, Output],
                [SelectionRangeSlider, Output],
                None,
            ),
            (
                ["provenance", "invalid_1"],
                ["year", "invalid_2"],
                [],
                [SelectionRangeSlider, Output],
                pytest.raises(ValueError),
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
        expected_cont_widget_types,
        expected_cat_widget_types,
        exception,
    ):
        expected_widget_types = [HBox, HBox, HBox, Button, DownloadButton]

        visualizer = JScatterVisualizer(
            [corpus],
            categorical_fields=cat_fields,
            continuous_filter_fields=cont_fields,
        )
        with exception or does_not_raise():
            visualizer.visualize()

            widgets = mock_display.call_args.args

            categorical_filters = widgets[1].children
            continous_filters = widgets[2].children

            assert [type(w) for w in widgets] == expected_widget_types
            assert [type(w) for w in continous_filters] == expected_cont_widget_types
            assert [type(w) for w in categorical_filters] == expected_cat_widget_types

    @pytest.mark.parametrize(
        "tooltip_fields,expected",
        [
            (["provenance"], {"provenance"}),
            (["provenance", "date"], {"provenance"}),
            (["provenance", "unknown"], {"provenance"}),
        ],
    )
    def test_valid_tooltip_fields(self, corpus, tooltip_fields, expected):
        visualizer = JScatterVisualizer([corpus], tooltip_fields=tooltip_fields)

        assert visualizer._tooltip_fields == expected

    @pytest.mark.skip(reason="TODO")
    def test_cluster_button(self, mock_display, corpus):
        visualizer = JScatterVisualizer([corpus])
        visualizer.visualize()

        widgets = mock_display.call_args.args

        assert visualizer.clusters is None

        widgets[-1].click()
        assert isinstance(visualizer._cluster_plot, JScatterVisualizer.PlotWidgets)
        assert all(isinstance(c, Corpus) for c in visualizer.clusters)
