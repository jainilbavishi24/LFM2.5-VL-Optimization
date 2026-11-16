"""GPU energy / power measurement via NVML.

Primary source: the cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ, Volta+).
Secondary: a background thread sampling power, SM clock and temperature, used for
average/peak power, to detect throttling, and as a fallback when the counter is missing.

Short intervals (< ~100 ms) are below the counter's update granularity, so the benchmark
also reports energy over all repeats of a configuration divided by the repeat count.

Jetson has no NVML: there the module input power (INA3221 rail VDD_IN, volts x amps in sysfs) is sampled every
5 ms by a background thread and integrated into the same kind of cumulative counter (JetsonPower), so every script
measures energy unchanged. Note: VDD_IN is the whole module (CPU + GPU + memory), not the GPU alone as with NVML.
"""

import glob
import os
import threading
import time

try:
    import pynvml as nv
except ImportError:  # pragma: no cover
    nv = None


def _read(path, default=0):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return default


class JetsonPower:
    """Jetson module power from the INA3221 rail VDD_IN, integrated by a background thread into a cumulative mJ
    counter (NVML-like), plus the GPU clock (devfreq) and temperature (thermal zone)."""

    def __init__(self, rail="VDD_IN", interval_s=0.005):
        self.v = self.i = None
        for lab in glob.glob("/sys/class/hwmon/hwmon*/in*_label"):
            try:
                if open(lab).read().strip() == rail:
                    d, n = os.path.dirname(lab), os.path.basename(lab)[2:-6]
                    self.v, self.i = f"{d}/in{n}_input", f"{d}/curr{n}_input"
            except OSError:
                pass
        if self.v is None:
            raise RuntimeError("no INA3221 rail " + rail)
        self.clk = next(iter(glob.glob("/sys/class/devfreq/*.gpu/cur_freq")), None)
        self.temp = next((os.path.join(os.path.dirname(t), "temp") for t in glob.glob("/sys/class/thermal/thermal_zone*/type")
                          if open(t).read().strip() == "gpu-thermal"), None)
        self.mj, self.interval = 0.0, interval_s
        self._last = (time.perf_counter(), self.power_w())
        threading.Thread(target=self._loop, daemon=True).start()

    def power_w(self):
        return _read(self.v) * _read(self.i) / 1e6                        # mV * mA -> W

    def _loop(self):
        while True:
            time.sleep(self.interval)
            t, p = time.perf_counter(), self.power_w()
            t0, p0 = self._last
            self.mj += (p + p0) / 2 * (t - t0) * 1e3                        # trapezoid, W*s -> mJ
            self._last = (t, p)

    def clock_mhz(self):
        return _read(self.clk) / 1e6 if self.clk else 0

    def temp_c(self):
        return _read(self.temp) / 1e3 if self.temp else 0


class EnergyMeter:
    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.01):
        self.available = False
        self.has_energy_counter = False
        self.sample_interval_s = sample_interval_s
        self.jetson = None
        self._thread = None
        self._samples = []
        if nv is not None:
            try:
                nv.nvmlInit()
                self.handle = nv.nvmlDeviceGetHandleByIndex(device_index)
                self.available = True
                try:
                    nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                    self.has_energy_counter = True
                    return
                except nv.NVMLError:
                    pass
            except Exception:
                pass
        try:  # Jetson: no NVML, or NVML without power/energy (JetPack 7 on Orin)
            self.jetson = JetsonPower()
            self.available = self.has_energy_counter = True
        except Exception:
            pass

    # ------------------------------------------------------------------ raw reads
    def energy_mj(self) -> int | None:
        if self.jetson is not None:
            return self.jetson.mj
        return nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.has_energy_counter else None

    def power_w(self) -> float:
        if self.jetson is not None:
            return self.jetson.power_w()
        return nv.nvmlDeviceGetPowerUsage(self.handle) / 1000

    def _clock_temp(self):
        if self.jetson is not None:
            return self.jetson.clock_mhz(), self.jetson.temp_c()
        return (nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM),
                nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU))

    def _sample_loop(self, stop: threading.Event):
        while not stop.is_set():
            try:
                self._samples.append((time.perf_counter(), self.power_w(), *self._clock_temp()))
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
