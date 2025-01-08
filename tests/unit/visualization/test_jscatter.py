import datetime
from contextlib import nullcontext as does_not_raise
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from ipywidgets.widgets import (
    BoundedIntText,
    Button,
    Checkbox,
    Dropdown,
    HBox,
    Label,
    SelectionRangeSlider,
    SelectMultiple,
    TagsInput,
    Text,
    VBox,
)

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.visualization.jscatter import (
    JScatterContainer,
    JScatterVisualizer,
)


@pytest.fixture
def mock_display():
    with mock.patch("tempo_embeddings.visualization.jscatter.display") as mock_display:
        yield mock_display


@pytest.fixture
def expected_main_widgets_no_cluster_button():
    """The main widgets *without the Cluster button*"""
    return [HBox, HBox, HBox, HBox, Dropdown, HBox, HBox, HBox, HBox]


@pytest.fixture
def expected_main_widgets(expected_main_widgets_no_cluster_button):
    """The main widgets *including the Cluster button*"""

    expected_main_widgets_no_cluster_button.insert(3, Button)
    return expected_main_widgets_no_cluster_button


class TestJScatterContainer:
    @pytest.fixture
    def container(self, corpus):
        return JScatterContainer([corpus])

    def test_init(self, expected_main_widgets, container):
        tab = container._tab

        assert [type(widget) for widget in tab.children] == [VBox]
        assert tab.titles == ("Full Corpus",)
        assert [
            type(widget) for widget in tab.children[0].children
        ] == expected_main_widgets

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
        "continuous_fields,categorical_fields,expected_continuous_filters,expected_categorical_filters,exception",
        [
            (["year"], ["provenance"], 1, 0, None),
            ([], ["provenance"], 0, 0, None),
            (["year"], [], 1, 0, None),
            (["year"], ["provenance", "invalid"], 1, 0, pytest.raises(ValueError)),
            (["year", "invalid"], ["provenance"], 1, 0, pytest.raises(ValueError)),
        ],
    )
    def test_get_widgets(
        self,
        corpus,
        expected_main_widgets_no_cluster_button,
        continuous_fields: list[str],
        categorical_fields: list[str],
        expected_continuous_filters: int,
        expected_categorical_filters: int,
        exception,
    ):
        visualizer = JScatterVisualizer(
            [corpus],
            continuous_fields=continuous_fields,
            categorical_fields=categorical_fields,
        )
        with exception or does_not_raise():
            top_widgets = visualizer.get_widgets()

            continous_filters = top_widgets[1].children
            categorical_filters = top_widgets[2].children

            assert [
                type(w) for w in top_widgets
            ] == expected_main_widgets_no_cluster_button
            assert [type(w) for w in continous_filters] == [
                SelectionRangeSlider
            ] * expected_continuous_filters
            assert [type(w) for w in categorical_filters] == [
                SelectMultiple
            ] * expected_categorical_filters

    @pytest.mark.parametrize(
        "tooltip_fields,expected",
        [
            (["provenance"], ["provenance"]),
            (["provenance", "date"], ["provenance"]),
            (["provenance", "unknown"], ["provenance"]),
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


class TestPlotWidgets:
    @pytest.fixture
    def plot_widgets(self, corpus):
        return JScatterVisualizer.PlotWidgets(
            df=corpus.to_dataframe().convert_dtypes(),
            color_by="corpus",
            tooltip_fields=set(),
        )

    def test_export_button(self, plot_widgets, tmp_path):
        expected_columns = [
            "text",
            "ID_DB",
            "highlight_start",
            "highlight_end",
            "year",
            "date",
            "provenance",
            "x",
            "y",
            "corpus",
            "distance_to_centroid",
        ]

        export_button = plot_widgets._export_button()
        assert isinstance(export_button, HBox)

        assert [type(w) for w in export_button.children] == [Button, Text, Checkbox]

        button, textbox, checkbox = export_button.children

        target_file = tmp_path / "export.csv"
        textbox.value = str(target_file)
        button.click()

        df = pd.read_csv(target_file)
        assert df.columns.to_list() == expected_columns
        assert len(df) == 5

    def test_color_by(self, plot_widgets):
        color_box = plot_widgets._color_by_dropdown()
        assert isinstance(color_box, Dropdown)
        assert color_box.value == "corpus"

        color_box.value = "year"
        assert plot_widgets._scatter_plot.color()["by"] == "year"

    def test_select_tooltips(self, plot_widgets):
        select_tooltips = plot_widgets._select_tooltips()
        assert isinstance(select_tooltips, HBox)

        label, tags_input = select_tooltips.children
        assert isinstance(label, Label)
        assert isinstance(tags_input, TagsInput)

        tags_input.value = ["provenance"]
        assert plot_widgets._scatter_plot.tooltip()["properties"] == ["provenance"]

    @pytest.mark.parametrize(
        "search_term,field,expected",
        [
            ("test", "text", np.arange(5)),
            ("test", "provenance", np.arange(5)),
            ("invalid", "text", []),
        ],
    )
    def test_search_filter(self, plot_widgets, search_term, field, expected):
        search_box = plot_widgets._search_filter()
        assert isinstance(search_box, HBox)
        assert [type(w) for w in search_box.children] == [Text, Dropdown]

        search_box.children[0].value = search_term
        search_box.children[1].value = field

        np.testing.assert_equal(plot_widgets._scatter_plot.filter(), expected)

        search_box.children[0].value = ""
        np.testing.assert_equal(plot_widgets._scatter_plot.filter(), np.arange(5))

    def test_plot_by_field_button(self, plot_widgets):
        button = plot_widgets._plot_by_field_button()
        assert isinstance(button, HBox)

        button, window_size_slider, groups_field_selector = button.children
        assert isinstance(button, Button)
        assert isinstance(window_size_slider, BoundedIntText)
        assert isinstance(groups_field_selector, Dropdown)

        with mock.patch("pandas.Series.plot") as mock_plot:
            button.click()
            mock_plot.assert_called_once_with(kind="line", legend=mock.ANY)

    def test_top_words_button(self, plot_widgets):
        keyword_extractor = mock.Mock()
        keyword_extractor.top_words.return_value = ["word1", "word2"]

        button = plot_widgets._top_words_button(keyword_extractor, umap_model=None)
        assert isinstance(button, HBox)

        button, text = button.children
        assert isinstance(button, Button)
        assert isinstance(text, Text)

        button.click()
        assert text.value == "word1; word2"
