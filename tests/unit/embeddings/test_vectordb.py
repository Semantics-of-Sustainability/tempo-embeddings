import pytest
import hashlib
from tempo_embeddings.embeddings.vector_database import ChromaDatabaseManager
from tempo_embeddings.text.passage import Passage


class TestChromaDB:

    @pytest.fixture()
    def my_passage(self) -> Passage:
        return Passage('de bouw zal dit jaar met tien  proeant verminderen.', 
                          metadata={'date': '09-07-1983', 'day': '9', 'filename': 
                                    'anp_1983_07_09_119_ocr.xml', 'issue': '119', 'month': '7', 
                        'tokenized_text': 'de bouw zal dit jaar met tien proeant verminderen .', 
                        'year': '1983'
                        })

    def test_passage_id(self, my_passage: Passage):
        mock_text = "de bouw zal dit jaar met tien  proeant verminderen."
        mock_meta = {'date': '09-07-1983', 'issue': '119', 'month': '7', 'day': '9', 'year': '1983',
                'filename': 'anp_1983_07_09_119_ocr.xml', 'tokenized_text': 'de bouw zal dit jaar met tien proeant verminderen .'}
        mock_highlighting = None
        meta_sorted = sorted(mock_meta.items(), key=lambda x: x[0])
        key = mock_text + str(meta_sorted) + str(mock_highlighting)
        mock_unique_id = hashlib.sha256(key.encode()).hexdigest()

        assert my_passage.get_unique_id() == mock_unique_id

    @pytest.fixture()
    def model_name(self) -> str:
        return "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"

    def test_create_new_invalid_db_noconfig(self, model_name: str):
        with pytest.raises(ValueError):
            ChromaDatabaseManager("pytest_db", model_name, embedder_config={"something": "wrong"})
    
    def test_create_new_invalid_db_nomodel(self, model_name: str):
        with pytest.raises(AttributeError):
            ChromaDatabaseManager("pytest_db", model_name, embedder_config={"type":"custom_model", "model": None})
    
    # def test_create_new_valid_db_custom_model(self, model_name: str):
    #     cfg = {"type": "custom_model", "model": SentenceTransformerModelWrapper.from_pretrained(model_name)}
    #     database = ChromaDatabaseManager("pytest_db", model_name, embedder_config=cfg)
    #     database.connect()

    # def test_create_valid_db_with_embedding_function(self, model_name: str):
    #     load_dotenv()
    #     cfg = {"type": "hf", "api_key": os.getenv('HF_API')}
    #     database = ChromaDatabaseManager("pytest_db_cm", model_name, embedder_config=cfg)
    #     database.connect()
    
    # @pytest.fixture()
    # def database(self, model_name: str):
    #     cfg = {"type": "custom_model", "model": SentenceTransformerModelWrapper.from_pretrained(model_name)}
    #     database = ChromaDatabaseManager("pytest_db", model_name, embedder_config=cfg)
    #     database.connect()
    #     return database

    # @pytest.fixture()
    # def collection(self, database: ChromaDatabaseManager):
    #     collection = database.create_new_collection("pytest")
    #     return collection