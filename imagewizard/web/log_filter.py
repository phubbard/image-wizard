"""Custom logging filters for the Uvicorn server."""

import logging


class InvalidHTTPFilter(logging.Filter):
    """Suppress the context-free 'Invalid HTTP request received.' warnings.

    These come from Uvicorn's H11 protocol layer when a client sends
    unparseable data (TLS handshakes on a plain-HTTP port, bot probes, etc.).
    The warnings carry no useful detail — the request was too broken to
    extract an IP, method, or path — so they just clutter the log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Invalid HTTP request received" in msg:
            return False
        return True
