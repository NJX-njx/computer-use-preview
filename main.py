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
import argparse
import os

from agent import BrowserAgent
from computers import BrowserbaseComputer, PlaywrightComputer
from data_pipeline.collector import CollectConfig, collect_episodes


PLAYWRIGHT_SCREEN_SIZE = (1440, 900)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser agent or collect early experience data.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("agent", "collect"),
        default="agent",
        help="Run mode: 'agent' to execute queries, 'collect' to gather early experience.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="The query for the browser agent to execute (agent mode only).",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=("playwright", "browserbase"),
        default="playwright",
        help="The computer use environment to use.",
    )
    parser.add_argument(
        "--initial_url",
        type=str,
        default="https://www.google.com",
        help="The inital URL loaded for the computer.",
    )
    parser.add_argument(
        "--highlight_mouse",
        action="store_true",
        default=False,
        help="If possible, highlight the location of the mouse.",
    )
    parser.add_argument(
        "--model",
        default='gemini-2.5-computer-use-preview-10-2025',
        help="Set which main model to use.",
    )
    parser.add_argument(
        "--use_behavior_prior",
        action="store_true",
        default=False,
        help="Enable behavior prior to validate/adjust actions during execution (agent mode).",
    )
    parser.add_argument(
        "--behavior_prior_ckpt",
        type=str,
        default="checkpoints/behavior_prior.json",
        help="Path to behavior prior checkpoint JSON (agent mode).",
    )
    # Collector specific flags
    parser.add_argument(
        "--collect_output_dir",
        type=str,
        default="data/early_experience",
        help="Directory to write collected episodes (collect mode).",
    )
    parser.add_argument(
        "--collect_episodes",
        type=int,
        default=2,
        help="Number of episodes to collect (collect mode).",
    )
    parser.add_argument(
        "--collect_max_steps",
        type=int,
        default=20,
        help="Max steps per episode (collect mode).",
    )
    parser.add_argument(
        "--collect_save_dom",
        action="store_true",
        default=False,
        help="Attempt to save DOM snapshots if supported (collect mode).",
    )
    parser.add_argument(
        "--collect_whitelist_domains",
        type=str,
        default="",
        help="Comma-separated domain whitelist; if set, stay within these domains (collect mode).",
    )
    parser.add_argument(
        "--collect_seed",
        type=int,
        default=0,
        help="Random seed (collect mode).",
    )
    args = parser.parse_args()

    if args.mode == "collect":
        def make_env():
            if args.env == "playwright":
                return PlaywrightComputer(
                    screen_size=PLAYWRIGHT_SCREEN_SIZE,
                    initial_url=args.initial_url,
                    highlight_mouse=args.highlight_mouse,
                )
            elif args.env == "browserbase":
                return BrowserbaseComputer(
                    screen_size=PLAYWRIGHT_SCREEN_SIZE,
                    initial_url=args.initial_url,
                )
            else:
                raise ValueError("Unknown environment: ", args.env)

        whitelist = (
            [d.strip() for d in args.collect_whitelist_domains.split(",") if d.strip()]
            if args.collect_whitelist_domains
            else None
        )
        cfg = CollectConfig(
            output_dir=args.collect_output_dir,
            episodes=args.collect_episodes,
            max_steps_per_episode=args.collect_max_steps,
            initial_url=args.initial_url,
            save_dom=args.collect_save_dom,
            whitelist_domains=whitelist,
            seed=args.collect_seed,
        )
        collect_episodes(make_env, cfg)
    else:
        if not args.query:
            raise SystemExit("--query is required in agent mode")
        if args.env == "playwright":
            env = PlaywrightComputer(
                screen_size=PLAYWRIGHT_SCREEN_SIZE,
                initial_url=args.initial_url,
                highlight_mouse=args.highlight_mouse,
            )
        elif args.env == "browserbase":
            env = BrowserbaseComputer(
                screen_size=PLAYWRIGHT_SCREEN_SIZE,
                initial_url=args.initial_url
            )
        else:
            raise ValueError("Unknown environment: ", args.env)

        with env as browser_computer:
            agent = BrowserAgent(
                browser_computer=browser_computer,
                query=args.query,
                model_name=args.model,
                use_behavior_prior=args.use_behavior_prior,
                behavior_prior_path=args.behavior_prior_ckpt,
            )
            agent.agent_loop()
    return 0


if __name__ == "__main__":
    main()
