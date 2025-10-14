# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Serialized observation for a single step."""

    url: str
    timestamp_ms: int
    # Optional path to the image file on disk; images are stored as PNG
    image_path: str
    # Optional DOM snapshot path (if recorded)
    dom_path: Optional[str] = None


class ActionRecord(BaseModel):
    """Structured action description recorded by the collector."""

    name: Literal[
        "open_web_browser",
        "click_at",
        "hover_at",
        "type_text_at",
        "scroll_document",
        "scroll_at",
        "wait_5_seconds",
        "go_back",
        "go_forward",
        "search",
        "navigate",
        "key_combination",
        "drag_and_drop",
    ]
    args: Dict[str, Any] = Field(default_factory=dict)


class StepRecord(BaseModel):
    """One transition (s, a, s')."""

    step_index: int
    obs: Observation
    action: ActionRecord
    next_obs: Observation
    meta: Dict[str, Any] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    episode_id: str
    steps: list[StepRecord]
