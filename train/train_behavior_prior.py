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

import argparse
from pathlib import Path
from models.behavior_prior import BehaviorPrior


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a simple behavior prior from collected episodes")
    parser.add_argument("--episodes_dir", type=str, required=True, help="Path to data/early_experience directory")
    parser.add_argument("--screen_w", type=int, default=1440)
    parser.add_argument("--screen_h", type=int, default=900)
    parser.add_argument("--out", type=str, default="checkpoints/behavior_prior.json")
    args = parser.parse_args()

    bp = BehaviorPrior((args.screen_w, args.screen_h))
    bp.fit_from_episodes(args.episodes_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bp.save(str(out))
    print(f"Saved behavior prior to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
