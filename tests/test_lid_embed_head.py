import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from training.embed_head import SWEEP, HeadConfig, MLPHead, train_head  # noqa: E402
from training.runlog import read_metrics  # noqa: E402


def clusters(per_label: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated clusters for eng_Latn (0), ind_Latn (1) and zsm_Latn (2)."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 8) * 4
    x = np.concatenate([centres[i] + rng.normal(scale=0.3, size=(per_label, 8)) for i in range(3)])
    y = np.repeat(np.arange(3), per_label)
    return x.astype(np.float16), y.astype(np.int64)


def test_sweep_matches_the_spec_grid():
    assert len(SWEEP) == 8
    assert {(c.hidden, c.lr, c.dropout) for c in SWEEP} == {
        (h, lr, d) for h in (128, 512) for lr in (1e-3, 3e-4) for d in (0.1, 0.3)
    }
    assert SWEEP[0].max_epochs == 30 and SWEEP[0].patience == 3


def test_head_output_shape():
    assert MLPHead(8, 4, 12, 0.1)(torch.zeros(5, 8)).shape == (5, 12)


def test_head_learns_separable_clusters(tmp_path):
    config = HeadConfig(hidden=16, lr=1e-2, dropout=0.0, max_epochs=15, batch_size=32)
    final = train_head(config, clusters(40, 0), clusters(10, 1), tmp_path, "head-00", env={})
    assert final["val"]["per_label"]["zsm_Latn"]["f1"] == 1.0
    assert final["val"]["group_f1"]["malay_indonesian"] == 1.0
    assert (tmp_path / "head-00" / "model.pt").exists()
    first = read_metrics(tmp_path / "head-00")[0]
    assert set(first) >= {"train_loss", "val_loss", "val_macro_f1", "val_group_macro_f1", "val_group_f1"}


def test_early_stopping_after_patience_epochs_without_improvement(tmp_path):
    config = HeadConfig(hidden=4, lr=0.0, dropout=0.0, max_epochs=30, patience=3, batch_size=64)
    final = train_head(config, clusters(10, 0), clusters(5, 1), tmp_path, "head-01", env={})
    assert final["best_epoch"] == 1
    assert final["epochs_run"] == 4
    assert len(read_metrics(tmp_path / "head-01")) == 4
