"""What counts as the right source for a question (spec section 11.2).

Each question is about one topic. A topic is answered by its Wikipedia article in any language, so a
retrieved chunk from any of the ids listed for the topic counts as a hit; that is deliberate, since
the knowledge base has no Traditional Chinese and no romanized Hindi documents at all
(DATASHEET.md), and cross-language retrieval is the only way those questions can be answered.

The ids are Wikipedia page ids, which stay the same when an article is edited, so this file survives
a rebuild of the knowledge base. Ids the rebuilt index does not hold are reported by missing_from
instead of quietly scoring zero.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPICS_PATH = REPO_ROOT / "eval" / "topics.json"


class TopicError(ValueError):
    """A topic file the evaluation cannot use."""


def load_topics(path: Path = TOPICS_PATH) -> dict[str, list[str]]:
    topics = json.loads(path.read_text(encoding="utf-8"))
    for topic, ids in sorted(topics.items()):
        if not ids:
            raise TopicError(f"{topic}: no document ids")
        for doc_id in ids:
            if not doc_id.startswith("wikipedia:"):
                raise TopicError(f"{topic}: {doc_id} is not a Wikipedia document id")
    return topics


def missing_from(topics: dict[str, list[str]], indexed: set[str]) -> dict[str, list[str]]:
    """Topic documents the index does not hold, so no question about them can score a hit there."""
    return {
        topic: sorted(set(ids) - indexed) for topic, ids in sorted(topics.items()) if set(ids) - indexed
    }
