import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

from get_latest_stamp import get_latest_stamp

directory = "../benchmarks/"
latest_benchmark = get_latest_stamp(directory)

# Load JSON
with open(latest_benchmark, "r") as f:
    data = json.load(f)

benchmarks = data["benchmarks"]

# Group by benchmark family
series = defaultdict(list)

name_re = re.compile(r"^(.*?)/(\d+)$")

for b in benchmarks:
    name = b.get("name", "")
    m = name_re.match(name)
    if not m:
        continue

    family = m.group(1)
    family = (family
              .replace("BM_FourierTransform_BigExample<float, ", "")
              .replace("<float>>", "")
              .replace(">", ""))
    N = int(m.group(2))

    series[family].append({
        "N": N,
        "real_time": b["real_time"],
        "items_per_second": b.get("items_per_second"),
    })

# Real time vs N
plt.figure()

for family, points in series.items():
    points.sort(key=lambda p: p["N"])

    Ns = [p["N"] for p in points]
    times = [p["real_time"] for p in points]

    plt.plot(
        Ns,
        times,
        markersize=4,
        marker="o",
        linestyle="--",
        label=family
    )

plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.locator_params(axis='y', numticks=30)
plt.locator_params(axis='x', numticks=30)
plt.xlabel("FT size (N)")
plt.ylabel("Time per FT (µs)")
plt.title("FT Time Complexity")
plt.axis("equal")
plt.legend()
plt.grid(True)

plt.show()

# Throughput (items/s)
plt.figure()

for family, points in series.items():
    points.sort(key=lambda p: p["N"])

    Ns = [p["N"] for p in points]
    items = [p["items_per_second"] for p in points]

    plt.plot(
        Ns,
        items,
        markersize=4,
        marker="o",
        linestyle="--",
        label=family
    )

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("FT size (N)")
plt.ylabel("Items per second")
plt.locator_params(axis='x', numticks=20)
plt.title("FT Throughput")
plt.legend()
plt.grid(True)

plt.show()
