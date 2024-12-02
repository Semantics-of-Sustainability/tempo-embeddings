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
        "cat_fields,cont_fields,expected_cont,expected_cat,exception",
        [
            (["provenance"], ["year"], 0, 1, None),
            (["provenance"], [], 0, 0, None),
            ([], ["year"], 1, 1, None),
            (["provenance", "invalid"], ["year"], 0, 1, pytest.raises(ValueError)),
            (["provenance"], ["year", "invalid"], 0, 1, pytest.raises(ValueError)),
        ],
    )
    def test_visualize(
        self,
        mock_display,
        corpus,
        cat_fields: list[str],
        cont_fields: list[str],
        expected_cont: int,
        expected_cat: int,
        exception,
    ):
        widget_types = [HBox, HBox, HBox, Button, DownloadButton]
        cat_types = [SelectionRangeSlider, Output]
        cont_types = [SelectMultiple, Output]

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

            assert [type(w) for w in widgets] == widget_types
            assert [type(w) for w in categorical_filters] == cat_types * expected_cat
            assert [type(w) for w in continous_filters] == cont_types * expected_cont

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
