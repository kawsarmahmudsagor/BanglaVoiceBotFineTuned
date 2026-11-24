"""`speakers` step: pseudo-speaker clustering (spec 8.2).

Bengali.AI has no speaker ids. Part 1 embeds every candidate clip with ECAPA (`speaker_emb.parquet`, resumable);
part 2 clusters the embeddings (always recomputed) into `speakers.parquet` with names `<prefix>-<n>`, numbered
by cluster size descending. Above `MAX_DIRECT` clips, clustering goes through MiniBatchKMeans centroids first,
because plain agglomerative clustering is O(n^2) in memory.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ..workdir import StepFile, WorkDir
from ._runner import run_step

EMB_STEP = "speaker_emb"
STEP = "speakers"
VERSION = 1
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
EMB_DIM = 192
THRESHOLD = 0.35
PREFIX = "bai"
MAX_DIRECT = 20000
MAX_CENTROIDS = 20000
EMB_COLUMNS = ["id", "emb", "error"]

# An embedder maps a mono float32 16 kHz waveform to a 192-d vector.
Embedder = Callable[[np.ndarray], np.ndarray]
_embedder: Embedder | None = None


def make_embedder() -> Embedder:
    """The real ECAPA embedder (GPU when available). Imports torch/speechbrain lazily."""
    import torch

    try:
        from speechbrain.inference.speaker import EncoderClassifier  # speechbrain 1.x
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # speechbrain 0.5.x (pinned by pyannote-audio 3.3.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(source=MODEL, run_opts={"device": device})

    def embedder(y16k: np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(y16k, dtype="float32")).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_batch(wav).reshape(-1).cpu().numpy()

    return embedder


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = make_embedder()
    return _embedder


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative clustering, cosine distance, average linkage; clusters merge until distance > threshold."""
    from sklearn.cluster import AgglomerativeClustering

    x = np.asarray(embeddings, dtype="float64")
    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold)
    return model.fit_predict(x).astype(int)


def cluster_two_stage(embeddings: np.ndarray, threshold: float, *, max_direct: int = MAX_DIRECT) -> np.ndarray:
    """`cluster()` directly up to max_direct clips; otherwise k-means centroids are clustered and clips inherit."""
    x = np.asarray(embeddings, dtype="float64")
    n = len(x)
    if n <= max_direct:
        return cluster(x, threshold)
    from sklearn.cluster import MiniBatchKMeans

    k = min(MAX_CENTROIDS, n // 10)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096).fit(_l2(x))
    centroid_labels = cluster(_l2(km.cluster_centers_), threshold)
    return centroid_labels[km.labels_]


def name_clusters(ids: list[str], labels: np.ndarray, *, prefix: str = PREFIX) -> list[str]:
    """`<prefix>-<n>`, n=1 for the largest cluster; ties broken by the smallest member id."""
    members: dict[int, list[str]] = {}
    for i, lab in zip(ids, labels):
        members.setdefault(int(lab), []).append(i)
    order = sorted(members, key=lambda lab: (-len(members[lab]), min(members[lab])))
    names = {lab: f"{prefix}-{n}" for n, lab in enumerate(order, start=1)}
    return [names[int(lab)] for lab in labels]


def _row(row: dict) -> dict:
    from .dnsmos import _load_16k

    emb = np.asarray(get_embedder()(_load_16k(Path(row["src_path"]))), dtype="float32").reshape(-1)
    if emb.shape != (EMB_DIM,) or not np.isfinite(emb).all():
        raise ValueError(f"bad embedding shape {emb.shape}")
    return {"id": row["id"], "emb": emb.tolist()}


def embed(
    wd: WorkDir, *, force: bool = False, retry_errors: bool = False, only_profile: str | None = None,
    labeller: str | None = None,
) -> int:
    only_ids = None
    if only_profile:
        from ..profiles import load_profile
        from ..select import eligible, resolve_skips

        profile = load_profile(only_profile)
        # The speaker floor needs this step's output, so it is always skipped here (spec 5.2 step 2).
        skip, notes = resolve_skips(wd, profile, labeller, frozenset({"speaker"}))
        for note in notes:
            print(f"speakers: {note}")
        only_ids = set(eligible(wd, profile, labeller, skip_columns=skip)["id"])
    get_embedder()  # fail fast on a missing extra instead of recording an error on every row
    return run_step(
        wd, step=EMB_STEP, version=VERSION, columns=EMB_COLUMNS, row_fn=_row, input_cols=["id", "src_path"],
        dtypes={"emb": "object"}, workers=1, force=force, retry_errors=retry_errors,
        params={"model": MODEL}, only_ids=only_ids,
    )


def cluster_step(
    wd: WorkDir, *, threshold: float = THRESHOLD, prefix: str = PREFIX, max_direct: int = MAX_DIRECT
) -> int:
    """Cluster speaker_emb.parquet into speakers.parquet (always recomputed). Returns the number of speakers."""
    df = StepFile.read(wd.step_path(EMB_STEP), step=EMB_STEP, version=VERSION, params={"model": MODEL})
    df = df[df["error"].isna()].sort_values("id", kind="stable").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no embeddings to cluster")
    x = np.stack([np.asarray(e, dtype="float32") for e in df["emb"]])
    labels = cluster_two_stage(x, threshold, max_direct=max_direct)
    ids = list(df["id"])
    out = pd.DataFrame({"id": ids, "speaker": name_clusters(ids, labels, prefix=prefix), "error": None})
    out = out.astype({"error": "object"})
    StepFile.write(wd.step_path(STEP), out, step=STEP, version=VERSION, params={"threshold": threshold})
    return out["speaker"].nunique()


def _cmd(args: argparse.Namespace) -> int:
    from ..select import SelectError

    wd = WorkDir(args.work)
    try:
        embed(
            wd, force=args.force, retry_errors=args.retry_errors, only_profile=args.only_profile,
            labeller=args.labeller,
        )
        n = cluster_step(wd, threshold=args.threshold, prefix=args.prefix)
    except (SelectError, RuntimeError, ImportError) as e:
        print(f"speakers: {e}")
        return 1
    print(f"speakers: {n} pseudo-speakers")
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("speakers", help="pseudo-speaker clustering (needs the score extra)")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--force", action="store_true", help="recompute all embeddings")
    p.add_argument("--retry-errors", action="store_true", help="recompute embeddings whose error is set")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="cosine distance threshold for merging clusters")
    p.add_argument("--prefix", default=PREFIX, help="speaker name prefix")
    p.add_argument(
        "--only-profile", default=None, metavar="NAME",
        help="embed only clips passing this profile's other filters (spec 5.2 step 2)",
    )
    p.add_argument("--labeller", default=None, help="CER labeller spec for --only-profile; without it the cer filter is skipped")
    p.set_defaults(func=_cmd)
