import logging

import numpy as np
from bokeh.models.widgets.sliders import RangeSlider

from tempo_embeddings.visualization.bokeh import BokehInteractiveVisualizer


class TestBokehInteractiveVisualizer:
    def test_create_layout(self, corpus, caplog):
        corpus.embeddings_2d = np.random.rand(len(corpus), 2)
        visualizer = BokehInteractiveVisualizer(corpus)

        with caplog.at_level(logging.WARNING):
            layout = visualizer._create_layout()
            assert layout.children == [visualizer._figure]
            assert caplog.record_tuples == [
                (
                    "root",
                    logging.WARNING,
                    "Column 'year' not found in cluster 'TestCorpus'.",
                ),
                ("root", logging.WARNING, "No year data found. Skipping year slider."),
            ]

    def test_create_layout_years(self, corpus):
        year = 1999
        for passage in corpus.passages:
            passage.metadata["year"] = year

        visualizer = BokehInteractiveVisualizer(corpus)
        layout = visualizer._create_layout()
        assert layout.children[0] == visualizer._figure
        assert isinstance(layout.children[1], RangeSlider)
