#!/usr/bin/env python3
"""
Jetson tegrastats 모니터링 & W&B 로깅
- RAM
- CPU 6코어 평균 사용률
- GPU 사용률
- CPU 다이 온도
- GPU 다이 온도
- 총 전력(VDD_IN)
- 연산코어 전력(VDD_CPU_GPU_CV)
- 종료 시 총 모니터링 시간 출력 (시:분:초 + 초)
"""

import subprocess
import re
import time
import wandb
from datetime import datetime

wandb.init(project="mkg", name=datetime.now().strftime("monitor-%m-%d-%H-%M-%S-%f"))

interval = 1  # 초 단위 로깅 주기

metrics_hist = {
    "ram_used_mb": [],
    "cpu_usage_avg": [],
    "gpu_usage": [],
    "cpu_temp_c": [],
    "gpu_temp_c": [],
    "power_total_w": [],
    "power_core_w": [],
}


def parse_tegrastats_line(line: str):
    out = {}

    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        out["ram_used_mb"] = int(m.group(1))

    cpu_vals = re.findall(r"(\d+)%@", line)
    if cpu_vals:
        cpu_vals = list(map(int, cpu_vals))
        out["cpu_usage_avg"] = sum(cpu_vals) / len(cpu_vals)

    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        out["gpu_usage"] = int(m.group(1))

    m = re.search(r"cpu@([0-9.]+)C", line)
    if m:
        out["cpu_temp_c"] = float(m.group(1))

    m = re.search(r"gpu@([0-9.]+)C", line)
    if m:
        out["gpu_temp_c"] = float(m.group(1))

    m = re.search(r"VDD_IN (\d+)mW", line)
    if m:
        out["power_total_w"] = int(m.group(1)) / 1000.0

    m = re.search(r"VDD_CPU_GPU_CV (\d+)mW", line)
    if m:
        out["power_core_w"] = int(m.group(1)) / 1000.0

    return out


def format_hms(seconds: float) -> str:
    """초 단위를 H:MM:SS 문자열로 변환"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def monitor():
    start_time = time.time()  # 시작 시각 기록

    proc = subprocess.Popen(
        ["sudo", "tegrastats", "--interval", str(interval * 1000)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    try:
        for line in proc.stdout:
            stats = parse_tegrastats_line(line)
            if not stats:
                continue

            for k, v in stats.items():
                metrics_hist[k].append(v)

            agg = {}
            for k, vals in metrics_hist.items():
                if vals:
                    agg[f"{k}_avg"] = sum(vals) / len(vals)
                    agg[f"{k}_peak"] = max(vals)

            wandb.log({**stats, **agg})

    except KeyboardInterrupt:
        # 총 모니터링 시간 계산 후 출력
        total_time = time.time() - start_time
        print("\nMonitoring stopped.")
        print(
            f"Total monitoring time: {format_hms(total_time)} "
            f"({total_time:.1f} seconds)"
        )
        for k, v in agg.items():
            if "ram" in k:
                if "peak" in k:
                    v = v / 1024  # MB -> GB
                    print(f"{k}: {v:.1f}")
                else:
                    continue  # RAM 평균은 출력하지 않음
            elif "temp" in k:
                if "peak" in k:
                    continue  # 온도 피크는 출력하지 않음
                else:
                    print(f"{k}: {v:.1f}")
            elif "power_core" in k:
                continue  # 연산코어 전력은 출력하지 않음
            elif "power_total_w_avg" in k:
                # convert averge W to Wh
                v = v * (total_time) / 3600.0
                print(f"power_total_Wh: {v:.2f}")
            else:
                print(f"{k}: {v:.1f}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    monitor()


"""
09-21-2025 15:38:17 RAM 3645/7620MB (lfb 63x4MB) SWAP 236/3810MB (cached 0MB) CPU [1%@729,1%@729,4%@729,1%@729,0%@729,0%@729] EMC_FREQ 0%@204 GR3D_FREQ 0%@[305] NVDEC off NVJPG off NVJPG1 off VIC off OFA off APE 200 cpu@44.125C soc2@43.375C soc0@42.125C gpu@44.531C tj@44.531C soc1@43.156C VDD_IN 2957mW/2957mW VDD_CPU_GPU_CV 486mW/486mW VDD_SOC 1012mW/1012mW
"""
