import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("deliveries.csv")
df.columns = df.columns.str.strip()

# -----------------------------
# Fix team names
# -----------------------------
team_mapping = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Super Giants",
    "Rising Pune Supergiants": "Rising Pune Super Giants",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad"
}

df["batting_team"] = df["batting_team"].replace(team_mapping)
df["bowling_team"] = df["bowling_team"].replace(team_mapping)

valid_teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]

df = df[df["batting_team"].isin(valid_teams)]
df = df[df["bowling_team"].isin(valid_teams)]

# Use first innings for score analysis
first_innings = df[df["inning"] == 1].copy()

print("\n========== IPL EDA INSIGHTS ==========\n")

# -----------------------------
# 1. Score Distribution
# -----------------------------
final_scores = first_innings.groupby("match_id")["total_runs"].sum()

plt.figure(figsize=(8, 5))
plt.hist(final_scores, bins=25, edgecolor="black")
plt.axvline(final_scores.mean(), linestyle="--", label=f"Average: {final_scores.mean():.0f}")
plt.axvline(180, linestyle="--", label="Strong Score: 180")
plt.title("IPL First Innings Score Distribution")
plt.xlabel("Final First Innings Score")
plt.ylabel("Number of Matches")
plt.legend()
plt.tight_layout()
plt.savefig("eda_1_score_distribution.png")
plt.close()

print("1. Score Distribution")
print("Average first innings score:", round(final_scores.mean(), 2))
print("Highest first innings score:", int(final_scores.max()))
print("Lowest first innings score:", int(final_scores.min()))
print("Insight: This shows what is a normal and strong IPL first-innings score.\n")

# -----------------------------
# 2. Average Score by Team
# -----------------------------
team_scores = (
    first_innings.groupby(["match_id", "batting_team"])["total_runs"]
    .sum()
    .reset_index()
)

team_avg = team_scores.groupby("batting_team")["total_runs"].mean().sort_values()

plt.figure(figsize=(10, 6))
team_avg.plot(kind="barh")
plt.title("Average First Innings Score by Team")
plt.xlabel("Average Score")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("eda_2_team_average_score.png")
plt.close()

print("2. Team Strength")
print(team_avg.sort_values(ascending=False))
print("Insight: Teams with higher average score have stronger batting performance historically.\n")

# -----------------------------
# 3. Powerplay Runs by Team
# -----------------------------
powerplay = first_innings[first_innings["over"] <= 6]

pp_scores = (
    powerplay.groupby(["match_id", "batting_team"])["total_runs"]
    .sum()
    .reset_index()
)

pp_avg = pp_scores.groupby("batting_team")["total_runs"].mean().sort_values()

plt.figure(figsize=(10, 6))
pp_avg.plot(kind="barh")
plt.title("Average Powerplay Runs by Team")
plt.xlabel("Average Runs in Powerplay")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("eda_3_powerplay_runs.png")
plt.close()

print("3. Powerplay Analysis")
print(pp_avg.sort_values(ascending=False))
print("Insight: Teams with higher powerplay scores usually get a stronger start.\n")

# -----------------------------
# 4. Death Overs Runs by Team
# -----------------------------
death_overs = first_innings[first_innings["over"] >= 15]

death_scores = (
    death_overs.groupby(["match_id", "batting_team"])["total_runs"]
    .sum()
    .reset_index()
)

death_avg = death_scores.groupby("batting_team")["total_runs"].mean().sort_values()

plt.figure(figsize=(10, 6))
death_avg.plot(kind="barh")
plt.title("Average Death Overs Runs by Team")
plt.xlabel("Average Runs in Overs 15-20")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("eda_4_death_overs_runs.png")
plt.close()

print("4. Death Overs Analysis")
print(death_avg.sort_values(ascending=False))
print("Insight: Death overs show finishing ability, which strongly affects final score.\n")

# -----------------------------
# 5. Wickets Lost vs Final Score
# -----------------------------
match_score = first_innings.groupby("match_id")["total_runs"].sum()
match_wickets = first_innings.groupby("match_id")["is_wicket"].sum()

wicket_df = pd.DataFrame({
    "wickets": match_wickets,
    "score": match_score
})

wicket_avg = wicket_df.groupby("wickets")["score"].mean()

plt.figure(figsize=(8, 5))
wicket_avg.plot(kind="bar")
plt.title("Wickets Lost vs Average Final Score")
plt.xlabel("Wickets Lost")
plt.ylabel("Average Final Score")
plt.tight_layout()
plt.savefig("eda_5_wickets_vs_score.png")
plt.close()

print("5. Wickets Impact")
print(wicket_avg)
print("Insight: Losing more wickets generally reduces final score.\n")

# -----------------------------
# 6. Run Rate vs Final Score
# -----------------------------
runrate_df = pd.DataFrame({
    "score": match_score,
    "balls": first_innings.groupby("match_id").size()
})

runrate_df["run_rate"] = (runrate_df["score"] * 6) / runrate_df["balls"]

runrate_df["run_rate_group"] = pd.cut(
    runrate_df["run_rate"],
    bins=[0, 6, 8, 10, 12, 20],
    labels=["0-6", "6-8", "8-10", "10-12", "12+"]
)

rr_avg = runrate_df.groupby("run_rate_group")["score"].mean()

plt.figure(figsize=(8, 5))
rr_avg.plot(kind="bar")
plt.title("Run Rate Range vs Average Final Score")
plt.xlabel("Run Rate Range")
plt.ylabel("Average Final Score")
plt.tight_layout()
plt.savefig("eda_6_run_rate_vs_score.png")
plt.close()

print("6. Run Rate Impact")
print(rr_avg)
print("Insight: Higher run rate leads to higher final score. This supports using run_rate in the model.\n")

# -----------------------------
# 7. Top 10 Run Scorers
# -----------------------------
top_batters = (
    df.groupby("batter")["batsman_runs"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
top_batters.plot(kind="barh")
plt.title("Top 10 Run Scorers in IPL Data")
plt.xlabel("Total Runs")
plt.ylabel("Batter")
plt.tight_layout()
plt.savefig("eda_7_top_batters.png")
plt.close()

print("7. Top Batters")
print(top_batters.sort_values(ascending=False))
print("Insight: Top run scorers show why player selection matters in prediction.\n")

# -----------------------------
# 8. Top 10 Wicket Takers
# -----------------------------
top_bowlers = (
    df[df["is_wicket"] == 1]
    .groupby("bowler")["is_wicket"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))
top_bowlers.plot(kind="barh")
plt.title("Top 10 Wicket Takers in IPL Data")
plt.xlabel("Total Wickets")
plt.ylabel("Bowler")
plt.tight_layout()
plt.savefig("eda_8_top_bowlers.png")
plt.close()

print("8. Top Bowlers")
print(top_bowlers.sort_values(ascending=False))
print("Insight: Wicket-taking bowlers affect opponent scoring and match outcome.\n")

# -----------------------------
# 9. Fours and Sixes by Team
# -----------------------------
boundaries = df[df["batsman_runs"].isin([4, 6])]

boundary_count = (
    boundaries.groupby(["batting_team", "batsman_runs"])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 6))
boundary_count.plot(kind="bar")
plt.title("Fours and Sixes by Team")
plt.xlabel("Team")
plt.ylabel("Boundary Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_9_boundaries_by_team.png")
plt.close()

print("9. Boundary Analysis")
print(boundary_count)
print("Insight: Boundary scoring reflects aggressive batting strength.\n")

print("========== EDA COMPLETED ==========")
print("Saved images:")
print("eda_1_score_distribution.png")
print("eda_2_team_average_score.png")
print("eda_3_powerplay_runs.png")
print("eda_4_death_overs_runs.png")
print("eda_5_wickets_vs_score.png")
print("eda_6_run_rate_vs_score.png")
print("eda_7_top_batters.png")
print("eda_8_top_bowlers.png")
print("eda_9_boundaries_by_team.png")