"""Append-only experiment logger with a machine-readable run manifest."""

import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, output_dir, metadata):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frames_path = os.path.join(output_dir, "frames.jsonl")
        self.events_path = os.path.join(output_dir, "events.jsonl")
        self._frames_file = open(self.frames_path, "w", encoding="utf-8")
        self._events_file = open(self.events_path, "w", encoding="utf-8")
        self.metadata = dict(metadata)
        self.metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
        self.frame_count = 0
        self._write_json("run_manifest.json", self.metadata)

    def log_frame(self, record):
        self._frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._frames_file.flush()
        self.frame_count += 1

    def log_event(self, record):
        self._events_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def write_summary(self, summary):
        self._write_json("metrics.json", summary)
        self._write_csv("metrics.csv", summary)

    def close(self):
        if not self._frames_file.closed:
            self._frames_file.close()
        if not self._events_file.closed:
            self._events_file.close()

    def _write_json(self, filename, content):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2)

    def _write_csv(self, filename, content):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("metric,value\n")
            for key, value in sorted(content.items()):
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                handle.write("{0},{1}\n".format(key, value))
