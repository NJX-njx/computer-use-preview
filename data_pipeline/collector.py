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

import os
import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from computers import Computer
from .schemas import Observation, ActionRecord, StepRecord, EpisodeRecord


@dataclass
class CollectConfig:
    output_dir: str
    episodes: int = 5
    max_steps_per_episode: int = 30
    initial_url: str = "https://www.google.com"
    save_dom: bool = False
    whitelist_domains: Optional[list[str]] = None
    seed: int = 0


def _ts_ms() -> int:
    return int(time.time() * 1000)


def _domain_allowed(url: str, whitelist: Optional[list[str]]) -> bool:
    if not whitelist:
        return True
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        return any(host.endswith(d) for d in whitelist)
    except Exception:
        return True


def _save_screenshot(ep_dir: Path, step_idx: int, png_bytes: bytes) -> str:
    img_path = ep_dir / f"step_{step_idx:06d}_obs.png"
    with open(img_path, "wb") as f:
        f.write(png_bytes)
    return str(img_path)


def _save_next_screenshot(ep_dir: Path, step_idx: int, png_bytes: bytes) -> str:
    img_path = ep_dir / f"step_{step_idx:06d}_next_obs.png"
    with open(img_path, "wb") as f:
        f.write(png_bytes)
    return str(img_path)


def _save_json(ep_dir: Path, name: str, data: dict) -> str:
    path = ep_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def _maybe_get_dom_snapshot(computer: Computer) -> Optional[str]:
    # Placeholder: current Computer interface doesn't provide DOM. Return None.
    return None


def _random_action(computer: Computer) -> ActionRecord:
    # Simple, safe-biased random policy. Coordinates are absolute pixels.
    acts = [
        "scroll_document",
        "scroll_at",
        "hover_at",
        "click_at",
        "wait_5_seconds",
        "go_back",
        "go_forward",
        "search",
    ]
    name = random.choice(acts)
    W, H = computer.screen_size()
    if name in ("hover_at", "click_at"):
        x = random.randint(10, max(10, W - 10))
        y = random.randint(80, max(80, H - 10))  # 避免过于靠上影响浏览器UI
        return ActionRecord(name=name, args={"x": x, "y": y})
    if name == "scroll_at":
        x = random.randint(10, max(10, W - 10))
        y = random.randint(120, max(120, H - 10))
        direction = random.choice(["up", "down"])  # 水平滚动较少见
        magnitude = random.choice([200, 400, 800])
        return ActionRecord(
            name=name, args={"x": x, "y": y, "direction": direction, "magnitude": magnitude}
        )
    if name == "scroll_document":
        return ActionRecord(name=name, args={"direction": random.choice(["up", "down"])})
    if name == "wait_5_seconds":
        return ActionRecord(name=name, args={})
    if name in ("go_back", "go_forward", "search"):
        return ActionRecord(name=name, args={})
    # Fallback
    return ActionRecord(name="wait_5_seconds", args={})


def collect_episodes(env_factory: Callable[[], Computer], cfg: CollectConfig) -> list[EpisodeRecord]:
    random.seed(cfg.seed)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    episodes_meta: list[EpisodeRecord] = []
    for epi in range(cfg.episodes):
        episode_id = str(uuid.uuid4())
        ep_dir = out_root / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        steps: list[StepRecord] = []
        with env_factory() as computer:
            # Ensure initial state
            state = computer.open_web_browser()
            obs_path = _save_screenshot(ep_dir, 0, state.screenshot)
            dom_path = _maybe_get_dom_snapshot(computer) if cfg.save_dom else None
            obs = Observation(url=state.url, timestamp_ms=_ts_ms(), image_path=obs_path, dom_path=dom_path)

            for t in range(1, cfg.max_steps_per_episode + 1):
                # Domain guard
                if not _domain_allowed(obs.url, cfg.whitelist_domains):
                    # Navigate back to initial URL to stay within whitelist
                    next_state = computer.navigate(cfg.initial_url)
                else:
                    # Sample and execute action
                    action = _random_action(computer)
                    try:
                        if action.name == "open_web_browser":
                            next_state = computer.open_web_browser()
                        elif action.name == "click_at":
                            next_state = computer.click_at(action.args["x"], action.args["y"])
                        elif action.name == "hover_at":
                            next_state = computer.hover_at(action.args["x"], action.args["y"])
                        elif action.name == "type_text_at":
                            next_state = computer.type_text_at(
                                x=action.args["x"],
                                y=action.args["y"],
                                text=action.args.get("text", ""),
                                press_enter=action.args.get("press_enter", False),
                                clear_before_typing=action.args.get("clear_before_typing", True),
                            )
                        elif action.name == "scroll_document":
                            next_state = computer.scroll_document(action.args["direction"])
                        elif action.name == "scroll_at":
                            next_state = computer.scroll_at(
                                x=action.args["x"],
                                y=action.args["y"],
                                direction=action.args["direction"],
                                magnitude=action.args["magnitude"],
                            )
                        elif action.name == "wait_5_seconds":
                            next_state = computer.wait_5_seconds()
                        elif action.name == "go_back":
                            next_state = computer.go_back()
                        elif action.name == "go_forward":
                            next_state = computer.go_forward()
                        elif action.name == "search":
                            next_state = computer.search()
                        elif action.name == "navigate":
                            next_state = computer.navigate(action.args["url"])
                        elif action.name == "key_combination":
                            next_state = computer.key_combination(action.args["keys"])  # list[str]
                        elif action.name == "drag_and_drop":
                            next_state = computer.drag_and_drop(
                                x=action.args["x"],
                                y=action.args["y"],
                                destination_x=action.args["destination_x"],
                                destination_y=action.args["destination_y"],
                            )
                        else:
                            next_state = computer.wait_5_seconds()
                    except Exception as e:
                        # On any execution error, wait a bit and continue to keep the episode alive
                        time.sleep(0.5)
                        next_state = computer.current_state()

                # Persist transition
                next_img_path = _save_next_screenshot(ep_dir, t, next_state.screenshot)
                next_dom_path = _maybe_get_dom_snapshot(computer) if cfg.save_dom else None
                next_obs = Observation(
                    url=next_state.url, timestamp_ms=_ts_ms(), image_path=next_img_path, dom_path=next_dom_path
                )

                step = StepRecord(
                    step_index=t,
                    obs=obs,
                    action=action if 'action' in locals() else ActionRecord(name='navigate', args={'url': cfg.initial_url}),
                    next_obs=next_obs,
                    meta={},
                )
                steps.append(step)

                # Prepare next loop
                obs = next_obs

        episode = EpisodeRecord(episode_id=episode_id, steps=steps)
        episodes_meta.append(episode)

        # Save episode index json
        _save_json(ep_dir, "episode.json", episode.model_dump())

    return episodes_meta
