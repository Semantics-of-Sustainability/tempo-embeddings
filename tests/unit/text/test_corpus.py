import logging
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestCorpus:
    def test_add(self, test_passages):
        expected = Corpus(test_passages[:2], None, umap_model=None)
        assert Corpus([test_passages[0]]) + Corpus([test_passages[1]]) == expected

    def test_add_umap_fitted(self, corpus):
        corpus._fit_umap()

        with pytest.raises(RuntimeError):
            corpus + Corpus([Passage("test")])

    @pytest.mark.parametrize(
        "passages, key, default_value, expected",
        [
            ([], "key", None, []),
            (
                [Passage("test text", metadata={"key": 1})],
                "key",
                None,
                [(1, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [Passage("test text", metadata={"key": 1})],
                "unknown key",
                None,
                [(None, [Passage("test text", metadata={"key": 1})])],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"key": 1}),
                ],
                "key",
                None,
                [
                    (
                        1,
                        [
                            Passage("test text 1", metadata={"key": 1}),
                            Passage("test text 2", metadata={"key": 1}),
                        ],
                    )
                ],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"key": 2}),
                ],
                "key",
                None,
                [
                    (1, [Passage("test text 1", metadata={"key": 1})]),
                    (2, [Passage("test text 2", metadata={"key": 2})]),
                ],
            ),
            (
                [
                    Passage("test text 1", metadata={"key": 1}),
                    Passage("test text 2", metadata={"other key": 1}),
                ],
                "key",
                0,
                [
                    (0, [Passage("test text 2", metadata={"other key": 1})]),
                    (1, [Passage("test text 1", metadata={"key": 1})]),
                ],
            ),
        ],
    )
    def test_groupby(self, passages, key, default_value, expected):
        groups = Corpus(passages).groupby(key, default_value=default_value)

        for (group, elements), (expected_group, expected_elements) in zip(
            groups, expected, **STRICT
        ):
            assert group == expected_group
            assert list(elements) == expected_elements

    @pytest.mark.parametrize(
        "n_passages,max_clusters,min_cluster_size",
        [(10, 10, 2), (50, 3, 2), (50, 10, 2), (50, 5, 5)],
    )
    def test_cluster(self, caplog, n_passages, max_clusters, min_cluster_size):
        corpus = Corpus(
            [
                Passage(f"test {str(i)}", embedding=np.random.rand(768))
                for i in range(n_passages)
            ]
        )

        with caplog.at_level(logging.WARNING):
            clusters = list(
                corpus.cluster(
                    max_clusters=max_clusters, min_cluster_size=min_cluster_size
                )
            )

        # FIXME: clustering is not deterministic
        assert (
            len(clusters) <= max_clusters
            or ("root", logging.WARNING, "Could not reduce number of clusters.")
            in caplog.record_tuples
        )

        for cluster in clusters:
            assert all(passage in corpus.passages for passage in cluster.passages)
            assert len(cluster) >= min_cluster_size or cluster._label == -1

    @pytest.mark.parametrize(
        "sample_size, centroid_based, exception",
        [(5, False, None), (0, False, None), (10, False, pytest.raises(ValueError))],
    )
    def test_sample(self, corpus, sample_size, centroid_based, exception):
        with exception or does_not_raise():
            sample = corpus.sample(sample_size, centroid_based=centroid_based)

            assert len(sample) == sample_size

            if centroid_based:
                # TODO: test that samples are closest to the centroid
                pass

    @pytest.mark.parametrize(
        "step_size,start, stop, expected",
        [
            (
                1,
                None,
                None,
                [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4), slice(4, 5)],
            ),
            (2, None, None, [slice(0, 2), slice(2, 4), slice(4, 5)]),
            (5, None, None, [slice(0, 5)]),
            (10, None, None, [slice(0, 5)]),
            (2, 1950, 1953, [slice(0, 2), slice(2, 3)]),
            (2, 1950, 1954, [slice(0, 2), slice(2, 4)]),
            (2, 1950, 1960, [slice(0, 2), slice(2, 4), slice(4, 5)]),
            (2, 1960, 1970, []),
        ],
    )
    def test_windows(self, corpus, step_size, start, stop, expected):
        expected_windows = [corpus.passages[_slice] for _slice in expected]
        windows = corpus.windows(step_size, start=start, stop=stop)

        for window, expected in zip(windows, expected_windows, **STRICT):
            assert window.passages == expected
            if stop is not None:
                assert int(window.label.split("-")[-1]) <= stop

    @pytest.mark.parametrize(
        "corpus,metadata_fields,expected",
        [
            (Corpus(), None, []),
            (Corpus([Passage("test")]), None, [{"text": "test", "corpus": "None"}]),
            (
                Corpus([Passage("test")], label="test label"),
                None,
                [{"text": "test", "corpus": "test label"}],
            ),
        ],
    )
    def test_hover_datas(self, corpus, metadata_fields, expected):
        assert corpus.hover_datas(metadata_fields=metadata_fields) == expected

    @pytest.mark.parametrize(
        "corpus,key,expected",
        [
            (Corpus(), "test key", []),
            (Corpus(passages=[Passage("text", metadata={"key": 1})]), "key", [1]),
            # FIXME: redundant test cases since passages are not filtered by highlighting
            (
                Corpus(
                    [
                        Passage(
                            "text",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        )
                    ],
                ),
                "key",
                [1],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"key": 2, "other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, 2],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, None],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage("text 2", metadata={"key": 2, "other": 2}),
                    ],
                ),
                "key",
                [1, 2],
            ),
        ],
    )
    def test_get_metadatas(self, corpus: Corpus, key, expected):
        assert list(corpus.get_metadatas(key)) == expected

    @pytest.mark.xfail(reason="Not implemented", raises=NotImplementedError)
    def test_load_save(self, tmp_path):
        # TODO: test/handle model (de)serialization
        filepath = tmp_path / "corpus"
        corpus = Corpus(
            [
                Passage("text 1", metadata={"key": 1, "other": 3}),
                Passage("text 2", metadata={"key": 2}),
            ],
            [Highlighting(0, 4)],
        )
        corpus.save(filepath)

        assert filepath.is_file()
        assert Corpus.load(filepath) == corpus

    @pytest.mark.parametrize(
        "corpus,expected",
        [
            (Corpus([Passage("test")]), np.array([None])),
            (Corpus([Passage("test", embedding=np.ones(10))]), np.ones((1, 10))),
            (
                Corpus(
                    [
                        Passage("test 1", embedding=np.ones(10)),
                        Passage("test 2", embedding=np.ones(10)),
                    ]
                ),
                np.ones((2, 10)),
            ),
        ],
    )
    def test_embeddings(self, corpus, expected):
        assert_equal(corpus.embeddings, expected)

    def test_select_embeddings(self, corpus):
        assert corpus._select_embeddings(use_2d_embeddings=False).shape == (5, 768)

        assert corpus._select_embeddings(use_2d_embeddings=True).shape == (5, 2)
        assert corpus._select_embeddings(use_2d_embeddings=False).shape == (5, 768)

    @pytest.mark.parametrize(
        "corpus",
        [
            (Corpus([Passage("test")])),
            (Corpus([Passage("test" + str(i)) for i in range(5)])),
        ],
    )
    def test_embeddings_setter(self, corpus):
        embeddings = [np.random.rand(10) for passage in corpus.passages]
        corpus.embeddings = np.array(embeddings)
        for passage, embedding in zip(corpus.passages, embeddings):
            assert_equal(passage.embedding, embedding)

    @pytest.mark.parametrize(
        "embeddings,expected",
        [
            (np.ones((1, 2)), pd.DataFrame({"x": [1], "y": [1]}, dtype=np.float64)),
            (None, pd.DataFrame()),
            (
                np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
                pd.DataFrame({"x": [1, 3, 5], "y": [2, 4, 6]}, dtype=np.float64),
            ),
        ],
    )
    def test_coordinates(self, corpus, embeddings, expected, caplog):
        if embeddings is not None:
            # set 2d embeddings
            corpus = corpus.sample(embeddings.shape[0])
            corpus.embeddings_2d = embeddings

        with caplog.at_level(logging.WARNING):
            pd.testing.assert_frame_equal(corpus.coordinates(), expected)

        if embeddings is None:
            expected_warning = ("root", logging.WARNING, "No 2D embeddings available.")
            assert expected_warning in caplog.record_tuples

    def test_centroid(self, corpus):
        corpus.embeddings = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float64
        )
        expected = np.array([5.0, 6.0], dtype=np.float64)

        np.testing.assert_equal(corpus.centroid(use_2d_embeddings=False), expected)

    @pytest.mark.parametrize(
        "embeddings,normalize,expected",
        [
            (
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                False,
                [0.026583, 0.001312, 0.0, 0.00029, 0.000725],
            ),
            (
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                True,
                [0.998354, 0.049287, 0.0, 0.010899, 0.027222],
            ),
        ],
    )
    def test_distances(self, corpus, embeddings, normalize, expected):
        corpus.embeddings = np.array(embeddings, dtype=np.float64)

        np.testing.assert_allclose(
            corpus.distances(normalize=normalize, use_2d_embeddings=False),
            np.array(expected, dtype=np.float64),
            rtol=1e-3,
        )

    def test_to_dataframe(self, corpus):
        ids = [
            "2ca2b35080541f54923e8672e8a6e8bbb8cdeb0ffbd3a9b2b0d9ea8666e830ba",
            "def9ccfb5acac052ad71b656aa47c0b6b707d7ac5c33e82ffb860f43518a0b79",
            "2c98aba242f96f5ffe4b4d5c79741b1135ba0a5b76c7c2741fee5353fbfbd690",
            "b0810559acde7d00598789cd9e3264174f24af0b0b84e081d4159691648c7a45",
            "c982e6944e0813cdc9a533f27b18edafee1c6fdc9da4d09175542444edcef99f",
        ]

        expected = pd.DataFrame(
            [
                {
                    "text": passage.text,
                    "ID_DB": _id,
                    "highlight_start": 1,
                    "highlight_end": 3,
                    "provenance": "test_file",
                    "year": str(year),
                    "x": 0.0,
                    "y": 0.0,
                    "distance_to_centroid": 0.0,
                }
                for passage, _id, year in zip(corpus.passages, ids, range(1950, 1956))
            ]
        )

        pd.testing.assert_frame_equal(corpus.to_dataframe(), expected, atol=100.0)

    @pytest.mark.parametrize(
        "corpus, expected_exception",
        [
            (Corpus(), pytest.raises(ValueError)),
            (Corpus([Passage("test text")]), pytest.raises(ValueError)),
            (Corpus([Passage(f"test text {str(i)}") for i in range(10)]), None),
        ],
    )
    def test_compress_embeddings(self, corpus: Corpus, expected_exception):
        corpus.embeddings = np.random.rand(len(corpus), 768)

        with expected_exception or does_not_raise():
            compressed = corpus.compress_embeddings()

        if expected_exception is None:
            assert compressed.shape == (len(corpus), 2)

            np.testing.assert_array_equal(corpus.embeddings_2d, compressed)

            assert corpus.embeddings.shape == (len(corpus), 768)

            if len(corpus) == 1:
                # For single samples, the output should be zeros
                assert_equal(compressed, np.zeros((len(corpus), 2)))

            assert all(passage.embedding_compressed for passage in corpus.passages)

    def test_embeddings_2d(self, corpus):
        assert corpus.embeddings_2d is None

        embeddings = np.random.rand(len(corpus), 2)
        corpus.embeddings_2d = embeddings

        for i, passage in enumerate(corpus.passages):
            np.testing.assert_equal(passage.embedding_compressed, embeddings[i, :])

        np.testing.assert_array_equal(corpus.embeddings_2d, embeddings)

    def test_fit_umap(self, corpus):
        assert not Corpus._is_fitted(corpus.umap)

        with pytest.raises(AttributeError):
            corpus.umap.transform(np.random.rand(10, 768))

        corpus._fit_umap()

        assert Corpus._is_fitted(corpus.umap)

        np.testing.assert_allclose(
            corpus.umap.transform(np.random.rand(10, 768)),
            np.zeros((10, 2)),
            atol=100.0,
        )

        with pytest.raises(RuntimeError):
            corpus._fit_umap()
