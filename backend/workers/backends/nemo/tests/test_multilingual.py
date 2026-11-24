"""Multilingual IndicConformer -> single-language conversion (pure helpers; numpy stands in for torch)."""
import io
import tarfile

import numpy as np
import pytest

from mlops_backend.multilingual import (
    monolingual_config, monolingual_state_dict, multilingual_langs, read_nemo_config,
)

LANGS = ["as", "bn", "hi"]
V, H = 4, 3


def _cfg():
    return {
        "tokenizer": {"type": "multilingual", "langs": {
            lg: {"dir": f"/cluster/{lg}", "type": "bpe", "model_path": f"nemo:{lg}_tokenizer.model",
                 "vocab_path": f"nemo:{lg}_vocab.txt", "spe_tokenizer_vocab": f"nemo:{lg}_tokenizer.vocab"}
            for lg in LANGS}},
        "train_ds": {"manifest_filepath": "/cluster/train.json", "return_language_id": True, "batch_size": 8},
        "validation_ds": {"manifest_filepath": ["/cluster/a.json"], "return_language_id": True},
        "decoder": {"_target_": "RNNTDecoder", "multisoftmax": True, "vocab_size": 12, "blank_as_pad": True},
        "joint": {"_target_": "RNNTJoint", "multilingual": True, "language_keys": LANGS, "num_classes": 12},
        "aux_ctc": {"ctc_loss_weight": 0.3, "decoder": {"num_classes": 12, "multisoftmax": True}},
        "labels": ["x"] * 12,
        "target": "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel",
    }


def test_multilingual_langs_keeps_tokenizer_order():
    assert multilingual_langs(_cfg()) == LANGS
    assert multilingual_langs({"tokenizer": {"type": "bpe", "dir": "x"}}) is None
    assert multilingual_langs({}) is None


def test_monolingual_config_strips_fork_keys_and_cluster_paths():
    src = _cfg()
    out = monolingual_config(src, "bn", "/tmp/tok")
    assert out["tokenizer"] == {"dir": "/tmp/tok", "type": "bpe"}
    assert "multisoftmax" not in out["decoder"] and "multisoftmax" not in out["aux_ctc"]["decoder"]
    assert "multilingual" not in out["joint"] and "language_keys" not in out["joint"]
    assert out["train_ds"]["manifest_filepath"] is None and "return_language_id" not in out["train_ds"]
    assert out["validation_ds"]["manifest_filepath"] is None
    assert out["train_ds"]["batch_size"] == 8 and "labels" not in out
    assert out["target"] == src["target"]
    assert src["tokenizer"]["type"] == "multilingual"  # input untouched


def test_monolingual_config_rejects_unknown_language():
    with pytest.raises(ValueError, match="'xx'"):
        monolingual_config(_cfg(), "xx", "/tmp/tok")


def _rows(n, start=0):
    return np.arange(start, start + n * H, dtype=np.float32).reshape(n, H)


def test_state_dict_slices_embedding_and_picks_language_heads():
    total = V * len(LANGS)
    emb = _rows(total + 1)
    sd = {
        "encoder.layers.0.weight": np.ones((2, 2)),
        "decoder.prediction.embed.weight": emb,
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": np.ones((8, H)),
        "joint.pred.weight": np.ones((H, H)),
        **{f"joint.joint_net.2.{lg}.weight": np.full((V + 1, H), i) for i, lg in enumerate(LANGS)},
        **{f"joint.joint_net.2.{lg}.bias": np.full((V + 1,), i) for i, lg in enumerate(LANGS)},
        "ctc_decoder.decoder_layers.0.weight": _rows(total + 1, 1000)[..., None],
        "ctc_decoder.decoder_layers.0.bias": np.arange(total + 1, dtype=np.float32),
    }
    out = monolingual_state_dict(sd, LANGS, "bn")
    want_emb = np.concatenate([emb[V:2 * V], emb[-1:]])
    assert np.array_equal(out["decoder.prediction.embed.weight"], want_emb)
    assert np.array_equal(out["joint.joint_net.2.weight"], np.full((V + 1, H), 1))
    assert np.array_equal(out["joint.joint_net.2.bias"], np.full((V + 1,), 1))
    assert not any(".as." in k or ".hi." in k or ".bn." in k for k in out)
    bias = out["ctc_decoder.decoder_layers.0.bias"]
    assert bias.tolist() == [4, 5, 6, 7, 12]  # bn tokens + the shared blank
    assert out["ctc_decoder.decoder_layers.0.weight"].shape == (V + 1, H, 1)
    assert out["encoder.layers.0.weight"] is sd["encoder.layers.0.weight"]
    assert out["decoder.prediction.dec_rnn.lstm.weight_ih_l0"] is sd["decoder.prediction.dec_rnn.lstm.weight_ih_l0"]


def test_state_dict_renames_per_language_ctc_heads():
    total = V * len(LANGS)
    sd = {"decoder.prediction.embed.weight": _rows(total + 1),
          **{f"ctc_decoder.decoder_layers.{lg}.0.weight": np.full((V + 1, H, 1), i) for i, lg in enumerate(LANGS)}}
    out = monolingual_state_dict(sd, LANGS, "hi")
    assert set(out) == {"decoder.prediction.embed.weight", "ctc_decoder.decoder_layers.0.weight"}
    assert np.array_equal(out["ctc_decoder.decoder_layers.0.weight"], np.full((V + 1, H, 1), 2))


def test_state_dict_rejects_uneven_vocabulary():
    with pytest.raises(ValueError, match="do not split"):
        monolingual_state_dict({"decoder.prediction.embed.weight": _rows(12)}, LANGS, "bn")


def test_read_nemo_config_with_dot_slash_members(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "m.nemo"
    data = b"tokenizer:\n  type: multilingual\n  langs:\n    bn: {type: bpe}\n"
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    assert multilingual_langs(read_nemo_config(path)) == ["bn"]
