from __future__ import annotations

import logging
import os
import typing as t
from dataclasses import asdict, dataclass
from functools import lru_cache, wraps

import requests

from ragas.utils import get_debug_mode

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

logger = logging.getLogger(__name__)


USAGE_TRACKING_URL = "https://t.explodinggradients.com"
RAGAS_DO_NOT_TRACK = "RAGAS_DO_NOT_TRACK"
RAGAS_DEBUG_TRACKING = "__RAGAS_DEBUG_TRACKING"
USAGE_REQUESTS_TIMEOUT_SEC = 1


@lru_cache(maxsize=1)
def do_not_track() -> bool:  # pragma: no cover
    # Returns True if and only if the environment variable is defined and has value True
    # The function is cached for better performance.
    # return os.environ.get(RAGAS_DO_NOT_TRACK, str(False)).lower() == "true"
    return True


@lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For BentoML developers only - debug and print event payload if turned on
    return os.environ.get(RAGAS_DEBUG_TRACKING, str(False)).lower() == "true"


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:  # pragma: no cover
    # Silent errors when tracking
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            if _usage_event_debugging():
                if get_debug_mode():
                    logger.error(
                        "Tracking Error: %s", err, stack_info=True, stacklevel=3
                    )
                else:
                    logger.info("Tracking Error: %s", err)
            else:
                logger.debug("Tracking Error: %s", err)

    return wrapper


@dataclass
class BaseEvent:
    event_type: str


@dataclass
class EvaluationEvent(BaseEvent):
    metrics: list[str]
    evaluation_mode: str
    num_rows: int


@silent
def track(event_properties: BaseEvent):
    if do_not_track():
        return

    payload = asdict(event_properties)

    if _usage_event_debugging():
        # For internal debugging purpose
        logger.info("Tracking Payload: %s", payload)
        return

    requests.post(USAGE_TRACKING_URL, json=payload, timeout=USAGE_REQUESTS_TIMEOUT_SEC)
