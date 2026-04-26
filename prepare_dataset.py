import pandas as pd

df = pd.read_csv("deliveries.csv")
df.columns = df.columns.str.strip()

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
df = df[df["inning"].isin([1, 2])].copy()

# -----------------------------
# Player historical stats
# -----------------------------
bat_stats = df.groupby("batter").agg(
    player_runs=("batsman_runs", "sum"),
    player_balls=("ball", "count"),
    player_fours=("batsman_runs", lambda x: (x == 4).sum()),
    player_sixes=("batsman_runs", lambda x: (x == 6).sum())
).reset_index()

bat_stats["batting_strike_rate"] = (
    bat_stats["player_runs"] * 100 / bat_stats["player_balls"]
).fillna(0)

bowl_stats = df.groupby("bowler").agg(
    bowler_runs_given=("total_runs", "sum"),
    bowler_balls=("ball", "count"),
    bowler_wickets=("is_wicket", "sum")
).reset_index()

bowl_stats["bowling_economy"] = (
    bowl_stats["bowler_runs_given"] * 6 / bowl_stats["bowler_balls"]
).fillna(0)

# -----------------------------
# Build match-level ML dataset
# One row = one team innings
# -----------------------------
rows = []

for (match_id, inning), inn in df.groupby(["match_id", "inning"]):
    batting_team = inn["batting_team"].iloc[0]
    bowling_team = inn["bowling_team"].iloc[0]

    final_score = inn["total_runs"].sum()
    wickets_lost = inn["is_wicket"].sum()
    total_balls = len(inn)

    batters = inn["batter"].dropna().unique()
    bowlers = inn["bowler"].dropna().unique()

    selected_bat_stats = bat_stats[bat_stats["batter"].isin(batters)]
    selected_bowl_stats = bowl_stats[bowl_stats["bowler"].isin(bowlers)]

    batting_strength = selected_bat_stats["batting_strike_rate"].mean()
    boundary_strength = (
        selected_bat_stats["player_fours"].sum() +
        selected_bat_stats["player_sixes"].sum() * 2
    )

    bowling_economy = selected_bowl_stats["bowling_economy"].mean()
    bowling_wicket_strength = selected_bowl_stats["bowler_wickets"].mean()

    run_rate = (final_score * 6 / total_balls) if total_balls > 0 else 0

    rows.append({
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_strength": batting_strength,
        "boundary_strength": boundary_strength,
        "bowling_economy": bowling_economy,
        "bowling_wicket_strength": bowling_wicket_strength,
        "wickets_lost": wickets_lost,
        "run_rate": run_rate,
        "final_score": final_score
    })

ml_df = pd.DataFrame(rows)
ml_df = ml_df.dropna()

ml_df.to_csv("ipl_ml_dataset.csv", index=False)

print("Created ipl_ml_dataset.csv")
print("Shape:", ml_df.shape)
print(ml_df.head())