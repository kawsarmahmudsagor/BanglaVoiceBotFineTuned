"""Hugging Face access pre-check (huggingface_hub imported lazily)."""
from __future__ import annotations

from app.core.prepare import PrepareError


def check_hf_access(repo: str) -> None:
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        auth_check(repo)
    except GatedRepoError as e:
        raise PrepareError(_msg(repo)) from e
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 401:
            raise PrepareError(_msg(repo)) from e
        raise


def _msg(repo: str) -> str:
    return f"no access to {repo}: accept its terms on Hugging Face and set HF_TOKEN"
