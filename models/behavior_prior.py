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

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ClickableActions = {"click_at", "hover_at"}
CoordActions = {"click_at", "hover_at", "scroll_at", "drag_and_drop"}


@dataclass
class ActionStats:
    count: int = 0
    # Per-axis mean/std for x, y in pixels
    mean_x: Optional[float] = None
    mean_y: Optional[float] = None
    std_x: Optional[float] = None
    std_y: Optional[float] = None
    # For scroll direction / magnitude
    direction_counts: Counter = None
    magnitudes: List[int] = None

    def to_dict(self):
        return {
            "count": self.count,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "std_x": self.std_x,
            "std_y": self.std_y,
            "direction_counts": dict(self.direction_counts or {}),
            "magnitudes": list(self.magnitudes or []),
        }

    @staticmethod
    def from_dict(d: dict) -> "ActionStats":
        st = ActionStats()
        st.count = d.get("count", 0)
        st.mean_x = d.get("mean_x")
        st.mean_y = d.get("mean_y")
        st.std_x = d.get("std_x")
        st.std_y = d.get("std_y")
        st.direction_counts = Counter(d.get("direction_counts", {}))
        st.magnitudes = list(d.get("magnitudes", []))
        return st


class BehaviorPrior:
    """A simple, data-driven prior for GUI actions using empirical stats.

    - Learns action type frequency.
    - For coordinate actions, learns per-axis mean/std and samples with Gaussian.
    - For scroll actions, learns direction/magnitude distributions.
    """

    def __init__(self, screen_size: Tuple[int, int]):
        self.screen_w, self.screen_h = screen_size
        self.action_counts: Counter = Counter()
        self.action_stats: Dict[str, ActionStats] = defaultdict(ActionStats)

    def fit_from_episodes(self, episodes_dir: str):
        """Reads episodes/<id>/episode.json files and aggregates statistics."""
        root = Path(episodes_dir)
        episodes_path = root / "episodes"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Episodes directory not found: {episodes_path}")

        for ep_dir in episodes_path.iterdir():
            if not ep_dir.is_dir():
                continue
            ep_file = ep_dir / "episode.json"
            if not ep_file.exists():
                continue
            with open(ep_file, "r", encoding="utf-8") as f:
                ep = json.load(f)
            for step in ep.get("steps", []):
                action = step.get("action", {})
                name = action.get("name")
                args = action.get("args", {})
                if not name:
                    continue
                self.action_counts[name] += 1
                st = self.action_stats[name]
                st.count += 1

                # Coordinates
                if name in CoordActions:
                    # Best-effort: use 'x','y' if present
                    x = args.get("x") or args.get("destination_x")
                    y = args.get("y") or args.get("destination_y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        # Incremental mean/std (Welford)
                        if st.mean_x is None:
                            st.mean_x, st.mean_y = float(x), float(y)
                            st.std_x, st.std_y = 0.0, 0.0
                            st._m2x, st._m2y, st._n = 0.0, 0.0, 1
                        else:
                            st._n += 1
                            dx = x - st.mean_x
                            st.mean_x += dx / st._n
                            st._m2x += dx * (x - st.mean_x)
                            dy = y - st.mean_y
                            st.mean_y += dy / st._n
                            st._m2y += dy * (y - st.mean_y)
                            st.std_x = math.sqrt(max(st._m2x / max(st._n - 1, 1), 1.0))
                            st.std_y = math.sqrt(max(st._m2y / max(st._n - 1, 1), 1.0))

                # Scroll specifics
                if name in ("scroll_at", "scroll_document"):
                    if st.direction_counts is None:
                        st.direction_counts = Counter()
                    direction = args.get("direction")
                    if isinstance(direction, str):
                        st.direction_counts[direction] += 1
                    if name == "scroll_at":
                        if st.magnitudes is None:
                            st.magnitudes = []
                        mag = args.get("magnitude")
                        if isinstance(mag, int):
                            st.magnitudes.append(mag)

        # Finalize std if only one sample
        for name, st in self.action_stats.items():
            if st.count <= 1:
                if st.std_x is None:
                    st.std_x = max(self.screen_w * 0.2, 1.0)
                if st.std_y is None:
                    st.std_y = max(self.screen_h * 0.2, 1.0)

    def save(self, path: str):
        out = {
            "screen_size": [self.screen_w, self.screen_h],
            "action_counts": dict(self.action_counts),
            "action_stats": {k: v.to_dict() for k, v in self.action_stats.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "BehaviorPrior":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        size = d.get("screen_size", [1440, 900])
        bp = BehaviorPrior((size[0], size[1]))
        bp.action_counts = Counter(d.get("action_counts", {}))
        bp.action_stats = {k: ActionStats.from_dict(v) for k, v in d.get("action_stats", {}).items()}
        return bp

    def action_prob(self, name: str) -> float:
        total = sum(self.action_counts.values())
        if total == 0:
            return 1.0 / 1.0
        return self.action_counts.get(name, 0) / total

    def sample_coordinates(self, name: str) -> Tuple[int, int]:
        st = self.action_stats.get(name)
        if not st or st.mean_x is None:
            x = self.screen_w // 2
            y = self.screen_h // 2
        else:
            sigma_x = st.std_x or (self.screen_w * 0.2)
            sigma_y = st.std_y or (self.screen_h * 0.2)
            x = int(random.gauss(st.mean_x, sigma_x))
            y = int(random.gauss(st.mean_y, sigma_y))
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))
        return x, y

    def sample_scroll(self, at: bool = False) -> Tuple[str, int]:
        name = "scroll_at" if at else "scroll_document"
        st = self.action_stats.get(name)
        # Direction
        if st and st.direction_counts:
            dirs, counts = zip(*st.direction_counts.items())
            total = sum(counts)
            r = random.randint(1, total)
            cum = 0
            chosen = dirs[0]
            for d, c in zip(dirs, counts):
                cum += c
                if r <= cum:
                    chosen = d
                    break
        else:
            chosen = random.choice(["up", "down"]) if not at else random.choice(["up", "down", "left", "right"])

        # Magnitude (only for scroll_at)
        if at:
            if st and st.magnitudes:
                mag = random.choice(st.magnitudes)
            else:
                mag = random.choice([200, 400, 800])
        else:
            mag = 0
        return chosen, mag

    def validate_or_adjust(self, name: str, args: dict) -> dict:
        """Validate LLM-proposed action args; adjust if out-of-bounds or missing.

        For coordinate actions, if (x,y) absent or out of screen, sample from prior.
        For scroll params, fill defaults using prior distributions.
        """
        out = dict(args) if args else {}
        if name in CoordActions:
            x, y = out.get("x"), out.get("y")
            need = False
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                need = True
            else:
                if x < 0 or y < 0 or x >= self.screen_w or y >= self.screen_h:
                    need = True
            if need:
                sx, sy = self.sample_coordinates(name)
                out["x"], out["y"] = sx, sy

        if name in ("scroll_at", "scroll_document"):
            direction = out.get("direction")
            mag = out.get("magnitude")
            dflt_dir, dflt_mag = self.sample_scroll(at=(name == "scroll_at"))
            if direction not in {"up", "down", "left", "right"}:
                out["direction"] = dflt_dir
            if name == "scroll_at" and not isinstance(mag, int):
                out["magnitude"] = dflt_mag
        return out
