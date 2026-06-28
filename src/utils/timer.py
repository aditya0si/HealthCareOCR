import time

class PipelineTimer:
    def __init__(self):
        self.starts = {}
        self.elapsed = {}

    def start(self, stage: str):
        """
        Starts timing a stage.
        """
        self.starts[stage] = time.time()

    def stop(self, stage: str):
        """
        Stops timing a stage and records the elapsed time.
        """
        if stage in self.starts:
            self.elapsed[stage] = time.time() - self.starts[stage]

    def get_summary(self) -> dict:
        """
        Returns a dictionary of stage names and their elapsed times in seconds.
        """
        return self.elapsed

    def print_summary(self):
        """
        Prints the elapsed times for all timed stages.
        """
        print("\n=== PIPELINE TIMING SUMMARY ===")
        total_time = 0.0
        for stage, duration in self.elapsed.items():
            print(f"  {stage:<30}: {duration:.3f} s")
            total_time += duration
        print(f"  {'TOTAL PIPELINE LATENCY':<30}: {total_time:.3f} s")
        print("===============================\n")

    def __call__(self, stage: str):
        class Context:
            def __init__(self, timer, stage_name):
                self.timer = timer
                self.stage_name = stage_name
            def __enter__(self):
                self.timer.start(self.stage_name)
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.timer.stop(self.stage_name)
        return Context(self, stage)
