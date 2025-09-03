import json
from collections import deque
import os
import random

#Todo - der muss noch angepasst werden
class PromptBalancer:
    def __init__(self, edit_library, rng, config, project_root):
        self.edit_library = edit_library
        self.rng = rng
        self.config = config
        self.state_path = os.path.join(project_root, "logs/state.json")
        self.queues = {}
        self._load_state()

    def _build_queue_if_needed(self, known_or_unknown, category, kind):
        key = (known_or_unknown, category, kind)
        if key in self.queues:
            return

        prompts_all = list(self.edit_library[category][kind])
        subset_size = 25
        uses = 1 if known_or_unknown == "known" else 14
        strategy = "random"

        # Subset wählen
        if subset_size < len(prompts_all):
                subset = self.rng.sample(prompts_all, subset_size)
        else:
            subset = prompts_all

        bag = []
        for p in subset:
            bag.extend([p] * uses)
        self.rng.shuffle(bag)
        self.queues[key] = deque(bag)

    def next(self, known_or_unknown, category, kind):
        self._build_queue_if_needed(known_or_unknown, category, kind)
        key = (known_or_unknown, category, kind)
        q = self.queues[key]
        if not q:
            raise RuntimeError(f"Prompt budget exhausted for ({known_or_unknown}, {category}, {kind}).")
        prompt = q.popleft()
        # gelegentlich speichern
        if self.rng.random() < 0.05:
            self._save_state()
        return prompt

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            data = {f"{k[0]}::{k[1]}::{k[2]}": list(v) for k, v in self.queues.items()}
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, bag in data.items():
                    split, cat, kind = k.split("::", 2)
                    self.queues[(split, cat, kind)] = deque(bag)
        except Exception:
            self.queues = {}
