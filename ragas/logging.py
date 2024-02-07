import json
import logging
import os
import traceback
from functools import wraps
from typing import Dict, Union

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", False) else logging.INFO, force=True
)

_logger = None


def getLogger():
    global _logger
    if not _logger:
        _logger = Logger()
    return _logger


class Logger(object):
    def __init__(self):
        self.log = logging.getLogger("ragas")
        self.awsRequestId = None
        self.awsXRayTraceId = None

    def setRequestId(self, id):
        self.awsRequestId = id

    def setXRayTraceId(self, traceId):
        self.awsXRayTraceId = traceId

    def _format(self, msg, args):
        for a in args:
            try:
                json.dumps(args[a])
            except TypeError:
                args[a] = str(args[a])
        m = {"message": msg}
        if self.awsRequestId:
            m["awsRequestId"] = self.awsRequestId
        if self.awsXRayTraceId:
            m["awsXRayTraceId"] = self.awsXRayTraceId
        m["args"] = args
        return json.dumps(m)

    def info(self, msg, **kwargs):
        self.log.info(self._format(msg, kwargs))

    def error(self, msg, **kwargs):
        self.log.error(self._format(msg, kwargs))

    def critical(self, msg, **kwargs):
        self.log.critical(self._format(msg, kwargs))

    def debug(self, msg, **kwargs):
        self.log.debug(self._format(msg, kwargs))

    def warning(self, msg, **kwargs):
        self.log.warning(self._format(msg, kwargs))

    def exception(self, e, **kwargs):
        msg = "\n".join(traceback.format_exception(type(e), e, tb=e.__traceback__))
        m = {"message": msg, "args": kwargs}
        self.log.error(self._format(msg, kwargs))
