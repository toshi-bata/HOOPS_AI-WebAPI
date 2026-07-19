"""Shared pytest configuration for the WebAPI test suite.

Demo-only endpoints (named similarity index management, embedding-model
switching, Shape Space Map, MFR/Part-Classification dataset browsing,
context-layer prediction) are gated behind HOOPS_AI_ENABLE_DEMO_FEATURES
(see core.require_demo_enabled). The test suite exercises these endpoints
directly, so enable the flag for the whole test session.
"""

import os

os.environ.setdefault("HOOPS_AI_ENABLE_DEMO_FEATURES", "true")
