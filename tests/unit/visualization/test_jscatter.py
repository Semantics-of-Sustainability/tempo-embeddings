import datetime
from contextlib import nullcontext as does_not_raise
from unittest import mock

import pandas as pd
import pytest
from ipywidgets.widgets import Button, HBox, SelectionRangeSlider, SelectMultiple, VBox

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.visualization.jscatter import (
    JScatterContainer,
    JScatterVisualizer,
)
from tempo_embeddings.visualization.util import DownloadButton


@pytest.fixture
def mock_display():
    with mock.patch("tempo_embeddings.visualization.jscatter.display") as mock_display:
        yield mock_display


class TestJScatterContainer:
    @pytest.fixture
    def container(self, corpus):
        return JScatterContainer([corpus])

    def test_init(self, container):
        tab = container._tab
        assert [type(widget) for widget in tab.children] == [VBox]
        assert tab.titles == ("Full Corpus",)

        assert [type(widget) for widget in tab.children[0].children] == [
            HBox,
            HBox,
            HBox,
            DownloadButton,
            Button,
            Button,
            Button,
        ]

    @pytest.mark.parametrize("title", [None, "Test Title"])
    def test_add_tab(self, container, corpus, title):
        container.add_tab(
            JScatterVisualizer([corpus], container=container), title=title
        )

        assert [type(widget) for widget in container._tab.children] == [VBox] * 2
        assert container._tab.titles == ("Full Corpus", title or "Clusters 1")
        assert container._tab.selected_index == 1

    def test_visualize(self, container, mock_display):
        container.visualize()

        mock_display.assert_called_once()


class TestJScatterVisualizer:
    @pytest.mark.skip(reason="TODO: correct expected values")
    def test_init_df(self, test_passages):
        test_passages[0].embedding_compressed = [1.1, 2.2]
        del test_passages[0].metadata["year"]  # test automatic filling of "year" field

        corpus = Corpus(test_passages[:1], label="TestCorpus")

        expected_df = pd.DataFrame(
            [
                {
                    "index": 0,
                    "text": "test text 0",
                    "ID_DB": "0f530c9dc158fa3617bbba2cc4608a1787f5c6d3511c31f63e30b52559ef6984",
                    "highlight_start": 1,
                    "highlight_end": 3,
                    "year": 1950,
                    "date": datetime.date(1950, 1, 1),
                    "provenance": "test_file",
                    "x": 1.1,
                    "y": 2.2,
                    "corpus": "TestCorpus",
                    "distance_to_centroid": None,
                    "label": "TestCorpus",
                }
            ]
        ).convert_dtypes()

        pd.testing.assert_frame_equal(
            JScatterVisualizer([corpus])._df.convert_dtypes(), expected_df
        )

    def test_init_df_year(self):
        year = 1950
        corpus = Corpus(
            passages=[
                Passage(
                    text="test text",
                    metadata={"date": datetime.date(year, 1, 1)},
                    embedding_compressed=[1.1, 2.2],
                )
            ]
        )
        assert JScatterVisualizer([corpus])._df["year"][0] == year

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
    def test_get_widgets(
        self,
        corpus,
        cat_fields: list[str],
        cont_fields: list[str],
        expected_cont: int,
        expected_cat: int,
        exception,
    ):
        # No Cluster button due to a lack of container
        expected_widget_types = [HBox, HBox, HBox, DownloadButton, Button, Button]

        visualizer = JScatterVisualizer(
            [corpus],
            categorical_fields=cat_fields,
            continuous_fields=cont_fields,
        )
        with exception or does_not_raise():
            top_widgets = visualizer.get_widgets()

            categorical_filters = top_widgets[1].children
            continous_filters = top_widgets[2].children

            assert [type(w) for w in top_widgets] == expected_widget_types
            assert [type(w) for w in categorical_filters] == [
                SelectionRangeSlider
            ] * expected_cat
            assert [type(w) for w in continous_filters] == [
                SelectMultiple
            ] * expected_cont

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

    def test_with_corpora(self, corpus):
        visualizer = JScatterVisualizer([corpus])

        new_visualizer = visualizer.with_corpora([corpus] * 2)

        for arg in (
            "_categorical_fields",
            "_continuous_fields",
            "_tooltip_fields",
            "_color_by",
            "_keyword_extractor",
        ):
            assert getattr(new_visualizer, arg) == getattr(visualizer, arg)

    @pytest.mark.skip(reason="TODO")
    def test_cluster_button(self, mock_display, corpus):
        visualizer = JScatterVisualizer([corpus])
        visualizer.visualize()

        widgets = mock_display.call_args.args

        assert visualizer.clusters is None

        widgets[-1].click()
        assert isinstance(visualizer._cluster_plot, JScatterVisualizer.PlotWidgets)
        assert all(isinstance(c, Corpus) for c in visualizer.clusters)
