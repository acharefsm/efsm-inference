import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("experiment_results.csv")

avg_df = df.groupby("fitness_type", as_index=False)[["levenshtein_distance", "total_wrong"]].mean()

fig, ax1 = plt.subplots(figsize=(8, 5))

x = avg_df["fitness_type"]
ax1.plot(x, avg_df["levenshtein_distance"], marker="o", label="Avg Levenshtein Distance")
ax1.plot(x, avg_df["total_wrong"], marker="s", label="Avg Total Wrong")

ax1.set_xlabel("fitness_type")
ax1.set_ylabel("Average Value")
ax1.set_title("Average Total Wrong and Levenshtein Distance by fitness_type")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

########################

groups = [(1, 5), (3, 10), (7, 15)]

avg_data = []
for init_d, max_d in groups:
    subset = df[(df["max_init_depth"] == init_d) & (df["max_depth"] == max_d)]
    avg_lev = subset["levenshtein_distance"].mean()
    avg_wrong = subset["total_wrong"].mean()
    avg_data.append({"Depth Group": f"({init_d},{max_d})", "Avg Levenshtein": avg_lev, "Avg Total Wrong": avg_wrong})

avg_df = pd.DataFrame(avg_data)

fig, ax1 = plt.subplots(figsize=(8, 5))

x = avg_df["Depth Group"]
ax1.plot(x, avg_df["Avg Levenshtein"], marker="o", label="Avg Levenshtein Distance")
ax1.plot(x, avg_df["Avg Total Wrong"], marker="s", label="Avg Total Wrong")

ax1.set_xlabel("Depth Group (max_init_depth, max_depth)")
ax1.set_ylabel("Average Value")
ax1.set_title("Average Total Wrong and Levenshtein Distance by Depth Group")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
