import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tests.lid_torch_fixtures import tiny_encoder, tiny_tokenizer  # noqa: E402
from training.data.rows import Row  # noqa: E402
from training.embed_cache import cache_split, embed_texts, load_e5, load_split, mean_pool  # noqa: E402


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]


def test_embeddings_are_unit_length_and_keep_input_order():
    model, tokenizer = tiny_encoder(), tiny_tokenizer()
    texts = ["soft robot silicone air", "robot", "udara lunak"]
    vectors = embed_texts(model, tokenizer, texts, batch_size=2)
    assert vectors.shape == (3, 16)
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-3)
    alone = embed_texts(model, tokenizer, ["robot"], batch_size=1)
    assert np.allclose(vectors[1], alone[0], atol=1e-3)


def test_cache_round_trip(tmp_path):
    rows = [
        Row("soft robot", "eng_Latn", "openlid", "lti"),
        Row("robot lunak", "ind_Latn", "openlid", "lti", "crop"),
    ]
    assert cache_split(tiny_encoder(), tiny_tokenizer(), rows, tmp_path, "val_select", 8, "cpu") == 2
    x, y, synthetic = load_split(tmp_path, "val_select")
    assert x.dtype == np.float16 and x.shape == (2, 16)
    assert y.tolist() == [0, 1]
    assert synthetic.tolist() == [False, True]


def test_missing_weights_name_the_download_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models.py --e5-torch"):
        load_e5(tmp_path)
