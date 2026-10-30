import argparse, json, time
from pathlib import Path
from k6_common import finish_report
from k6_text import text_stage
from k6_sample import sample_stage
from k6_video import video_stage
from k6_audio import audio_stage
from k6_export import export_stage

STAGES = {"text": text_stage, "sample": sample_stage, "video": video_stage,
          "audio": audio_stage, "export": export_stage}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["text", "sample", "video", "audio", "export"], required=True)
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.config).read_text())
    started = time.perf_counter()
    measurements = {}
    STAGES[arguments.stage](configuration, measurements)
    finish_report(configuration, arguments.stage, measurements, started)
