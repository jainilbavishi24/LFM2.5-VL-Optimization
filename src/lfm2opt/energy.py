"""GPU energy / power measurement via NVML.

Primary source: the cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ, Volta+).
Secondary: a background thread sampling power, SM clock and temperature, used for
average/peak power, to detect throttling, and as a fallback when the counter is missing.

Short intervals (< ~100 ms) are below the counter's update granularity, so the benchmark
also reports energy over all repeats of a configuration divided by the repeat count.

Jetson has no NVML: energy there must come from tegrastats / INA3221 rails (TODO when we get the board).
"""

import threading
import time

try:
    import pynvml as nv
except ImportError:  # pragma: no cover
    nv = None


class EnergyMeter:
    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.01):
        self.available = False
        self.has_energy_counter = False
        self.sample_interval_s = sample_interval_s
        if nv is None:
            return
        try:
            nv.nvmlInit()
            self.handle = nv.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.has_energy_counter = True
            except nv.NVMLError:
                pass
        except Exception:
            pass
        self._thread = None
        self._samples = []

    # ------------------------------------------------------------------ raw reads
    def energy_mj(self) -> int | None:
        return nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.has_energy_counter else None

    def power_w(self) -> float:
        return nv.nvmlDeviceGetPowerUsage(self.handle) / 1000

    def _sample_loop(self, stop: threading.Event):
        while not stop.is_set():
            try:
                self._samples.append((
                    time.perf_counter(),
                    self.power_w(),
                    nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM),
                    nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU),
                ))
            except Exception:
                pass
            stop.wait(self.sample_interval_s)

    # ------------------------------------------------------------------ interval API
    def start(self):
        if not self.available:
            return
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, args=(self._stop,), daemon=True)
        self._e0 = self.energy_mj()
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> dict:
        """Returns energy (J), duration, and power/clock statistics for the interval."""
        if not self.available:
            return {}
        t1 = time.perf_counter()
        e1 = self.energy_mj()
        self._stop.set()
        self._thread.join()
        dur = t1 - self._t0
        powers = [s[1] for s in self._samples]
        clocks = [s[2] for s in self._samples]
        temps = [s[3] for s in self._samples]
        out = {"duration_s": dur, "n_power_samples": len(powers)}
        if e1 is not None and self._e0 is not None:
            out["energy_j"] = (e1 - self._e0) / 1000
        if powers:
            out["power_avg_w"] = sum(powers) / len(powers)
            out["power_max_w"] = max(powers)
            out["sm_clock_min_mhz"] = min(clocks)
            out["sm_clock_avg_mhz"] = sum(clocks) / len(clocks)
            out["temp_max_c"] = max(temps)
            if "energy_j" not in out:
                out["energy_j_from_samples"] = out["power_avg_w"] * dur
        return out

    def measure_idle(self, seconds: float = 5.0) -> dict:
        """Idle power baseline, so dynamic energy = energy - idle_power * time can be reported."""
        self.start()
        time.sleep(seconds)
        res = self.stop()
        if "energy_j" in res:
            res["idle_power_w"] = res["energy_j"] / res["duration_s"]
        elif "power_avg_w" in res:
            res["idle_power_w"] = res["power_avg_w"]
        return res
