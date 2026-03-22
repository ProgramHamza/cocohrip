import matplotlib.pyplot as plt

# Example data from your graph
videos = [
    "2026-02-25_17-27-27.mp4",
    "2026-02-25_17-29-20.mp4",
    "2026-02-25_17-32-05.mp4"
]
raw_success = [96.7, 100.0, 85.0]  # Replace with your actual raw values

plt.figure(figsize=(12, 6))
bars = plt.bar(videos, raw_success, color="#42a5f5", label="raw")

for i, b in enumerate(bars):
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{raw_success[i]:.1f}%", ha="center", va="bottom", fontsize=12)

plt.ylabel("Success rate (%)")
plt.title("OldBoardBetter success rate per video (first N frames): raw only")
plt.ylim(0, 105)
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.show()