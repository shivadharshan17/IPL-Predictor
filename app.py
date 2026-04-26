from flask import Flask, render_template_string, request
import pandas as pd
import joblib

app = Flask(__name__)

deliveries = pd.read_csv("deliveries.csv")
deliveries.columns = deliveries.columns.str.strip()

model = joblib.load("model_pipeline.pkl")

team_mapping = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Super Giants",
    "Rising Pune Supergiants": "Rising Pune Super Giants",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad"
}

deliveries["batting_team"] = deliveries["batting_team"].replace(team_mapping)
deliveries["bowling_team"] = deliveries["bowling_team"].replace(team_mapping)

teams = [
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

deliveries = deliveries[
    deliveries["batting_team"].isin(teams) &
    deliveries["bowling_team"].isin(teams)
]

bat_stats = deliveries.groupby("batter").agg(
    runs=("batsman_runs", "sum"),
    balls=("ball", "count"),
    fours=("batsman_runs", lambda x: (x == 4).sum()),
    sixes=("batsman_runs", lambda x: (x == 6).sum())
).reset_index()

bat_stats["strike_rate"] = (bat_stats["runs"] * 100 / bat_stats["balls"]).fillna(0)

bowl_stats = deliveries.groupby("bowler").agg(
    runs_given=("total_runs", "sum"),
    balls=("ball", "count"),
    wickets=("is_wicket", "sum")
).reset_index()

bowl_stats["economy"] = (bowl_stats["runs_given"] * 6 / bowl_stats["balls"]).fillna(0)


def get_team_players(team):
    batters = set(deliveries[deliveries["batting_team"] == team]["batter"].dropna())
    bowlers = set(deliveries[deliveries["bowling_team"] == team]["bowler"].dropna())
    return sorted(batters.union(bowlers))


def get_default_players(team, exclude=None):
    exclude = set(exclude or [])
    players = get_team_players(team)
    scores = {}

    for player in players:
        bat_row = bat_stats[bat_stats["batter"] == player]
        bowl_row = bowl_stats[bowl_stats["bowler"] == player]

        runs = bat_row["runs"].iloc[0] if not bat_row.empty else 0
        wickets = bowl_row["wickets"].iloc[0] if not bowl_row.empty else 0

        scores[player] = runs + wickets * 25

    ranked = sorted(players, key=lambda p: scores.get(p, 0), reverse=True)

    selected = []
    for player in ranked:
        if player not in exclude and player not in selected:
            selected.append(player)
        if len(selected) == 11:
            break

    return selected


def calculate_features(batting_team, bowling_team, batting_players, bowling_players):
    selected_batters = bat_stats[bat_stats["batter"].isin(batting_players)]
    selected_bowlers = bowl_stats[bowl_stats["bowler"].isin(bowling_players)]

    batting_strength = selected_batters["strike_rate"].mean()
    boundary_strength = selected_batters["fours"].sum() + selected_batters["sixes"].sum() * 2
    bowling_economy = selected_bowlers["economy"].mean()
    bowling_wicket_strength = selected_bowlers["wickets"].mean()

    if pd.isna(batting_strength):
        batting_strength = 0
    if pd.isna(bowling_economy):
        bowling_economy = 8
    if pd.isna(bowling_wicket_strength):
        bowling_wicket_strength = 0

    expected_run_rate = batting_strength / 100 * 6
    if expected_run_rate <= 0:
        expected_run_rate = 7

    expected_wickets = max(1, min(9, int(bowling_wicket_strength % 10)))

    return pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_strength": batting_strength,
        "boundary_strength": boundary_strength,
        "bowling_economy": bowling_economy,
        "bowling_wicket_strength": bowling_wicket_strength,
        "wickets_lost": expected_wickets,
        "run_rate": expected_run_rate
    }])


def batting_table(players):
    table = bat_stats[bat_stats["batter"].isin(players)].copy()

    table = table.rename(columns={
        "batter": "Player",
        "runs": "Runs",
        "balls": "Balls",
        "fours": "Fours",
        "sixes": "Sixes",
        "strike_rate": "Strike Rate"
    })

    table["Strike Rate"] = table["Strike Rate"].round(2)

    return table.sort_values("Runs", ascending=False).head(11).to_dict("records")


def bowling_table(players):
    table = bowl_stats[bowl_stats["bowler"].isin(players)].copy()

    table = table.rename(columns={
        "bowler": "Player",
        "runs_given": "Runs Given",
        "balls": "Balls",
        "wickets": "Wickets",
        "economy": "Economy"
    })

    table["Economy"] = table["Economy"].round(2)

    return table.sort_values("Wickets", ascending=False).head(11).to_dict("records")


def head_to_head_insights(team1, team2):
    pair_df = deliveries[
        ((deliveries["batting_team"] == team1) & (deliveries["bowling_team"] == team2)) |
        ((deliveries["batting_team"] == team2) & (deliveries["bowling_team"] == team1))
    ].copy()

    if pair_df.empty:
        return {
            "matches": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "team1_win_pct": 0,
            "team2_win_pct": 0,
            "team1_avg_score": 0,
            "team2_avg_score": 0,
            "avg_margin": 0,
            "summary": "No historical head-to-head data available."
        }

    innings_scores = (
        pair_df.groupby(["match_id", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    team1_scores = innings_scores[innings_scores["batting_team"] == team1]["total_runs"]
    team2_scores = innings_scores[innings_scores["batting_team"] == team2]["total_runs"]

    team1_wins = 0
    team2_wins = 0
    margins = []

    for match_id in innings_scores["match_id"].unique():
        match = innings_scores[innings_scores["match_id"] == match_id]

        if team1 in match["batting_team"].values and team2 in match["batting_team"].values:
            score1 = match[match["batting_team"] == team1]["total_runs"].iloc[0]
            score2 = match[match["batting_team"] == team2]["total_runs"].iloc[0]

            if score1 > score2:
                team1_wins += 1
                margins.append(score1 - score2)
            elif score2 > score1:
                team2_wins += 1
                margins.append(score2 - score1)

    total_matches = team1_wins + team2_wins

    if total_matches > 0:
        team1_win_pct = round((team1_wins / total_matches) * 100, 2)
        team2_win_pct = round((team2_wins / total_matches) * 100, 2)
    else:
        team1_win_pct = 0
        team2_win_pct = 0

    if total_matches == 0:
        summary = "Complete match results could not be reconstructed from available deliveries data."
    elif team1_wins > team2_wins:
        summary = f"{team1} has won more matches against {team2} historically."
    elif team2_wins > team1_wins:
        summary = f"{team2} has won more matches against {team1} historically."
    else:
        summary = "Both teams have an equal head-to-head record."

    return {
        "matches": total_matches,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "team1_win_pct": team1_win_pct,
        "team2_win_pct": team2_win_pct,
        "team1_avg_score": round(team1_scores.mean(), 2) if not team1_scores.empty else 0,
        "team2_avg_score": round(team2_scores.mean(), 2) if not team2_scores.empty else 0,
        "avg_margin": round(sum(margins) / len(margins), 2) if margins else 0,
        "summary": summary
    }


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IPL Winner Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 1250px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0 0 18px rgba(0,0,0,0.1);
        }

        h1 {
            color: #16213e;
            font-size: 36px;
        }

        h2 {
            color: #243b55;
            margin-top: 30px;
        }

        h3 {
            color: #16213e;
        }

        .row {
            display: flex;
            gap: 25px;
            align-items: flex-start;
        }

        .col {
            flex: 1;
        }

        label {
            display: block;
            margin-top: 10px;
            margin-bottom: 5px;
            font-weight: 600;
        }

        select {
            width: 100%;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            border: 1px solid #ccc;
            background: #f8f9fb;
        }

        button {
            background: #1f6feb;
            color: white;
            padding: 14px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            margin-top: 25px;
            width: 100%;
        }

        button:hover {
            background: #0f4db8;
        }

        .insight {
            padding: 20px;
            background: #eef5ff;
            border-left: 6px solid #1f6feb;
            margin-top: 20px;
            border-radius: 8px;
        }

        .result {
            padding: 20px;
            background: #e8f5e9;
            border-left: 6px solid #2e7d32;
            margin-top: 25px;
            border-radius: 8px;
        }

        .error {
            padding: 15px;
            background: #ffebee;
            border-left: 6px solid #c62828;
            margin-top: 20px;
            border-radius: 8px;
            color: #b71c1c;
        }

        .score-box {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }

        .score-card {
            flex: 1;
            background: #f1f5ff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .score-card .score {
            font-size: 34px;
            font-weight: bold;
            color: #1f6feb;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 7px;
            text-align: left;
        }

        th {
            background: #16213e;
            color: white;
        }

        ul {
            background: #fff8e1;
            padding: 18px 35px;
            border-radius: 10px;
            border-left: 6px solid #ffb300;
        }

        @media (max-width: 900px) {
            .row {
                flex-direction: column;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <h1>🏏 IPL Match Winner Prediction</h1>

    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}

    <form method="POST">

        <div class="row">
            <div class="col">
                <label>Select Team 1</label>
                <select name="team1" onchange="this.form.submit()">
                    {% for team in teams %}
                        <option value="{{ team }}" {% if team == team1 %}selected{% endif %}>
                            {{ team }}
                        </option>
                    {% endfor %}
                </select>
            </div>

            <div class="col">
                <label>Select Team 2</label>
                <select name="team2" onchange="this.form.submit()">
                    {% for team in teams %}
                        <option value="{{ team }}" {% if team == team2 %}selected{% endif %}>
                            {{ team }}
                        </option>
                    {% endfor %}
                </select>
            </div>
        </div>

        <h2>📊 Team Head-to-Head Insights</h2>

        <div class="insight">
            <h3>{{ team1 }} vs {{ team2 }}</h3>

            <p><b>Total Matches Played:</b> {{ h2h.matches }}</p>
            <p><b>{{ team1 }} Wins:</b> {{ h2h.team1_wins }} ({{ h2h.team1_win_pct }}%)</p>
            <p><b>{{ team2 }} Wins:</b> {{ h2h.team2_wins }} ({{ h2h.team2_win_pct }}%)</p>
            <p><b>{{ team1 }} Average Score:</b> {{ h2h.team1_avg_score }}</p>
            <p><b>{{ team2 }} Average Score:</b> {{ h2h.team2_avg_score }}</p>
            <p><b>Average Winning Margin:</b> {{ h2h.avg_margin }} runs</p>
            <p><b>Insight:</b> {{ h2h.summary }}</p>
        </div>

        <h2>👥 Select Playing XI</h2>

        <div class="row">
            <div class="col">
                <h3>{{ team1 }}</h3>

                {% for i in range(11) %}
                    <label>Player {{ i + 1 }}</label>
                    <select name="team1_player_{{ i }}" onchange="this.form.submit()">
                        {% for player in team1_player_list %}
                            <option value="{{ player }}"
                                {% if team1_players|length > i and team1_players[i] == player %}selected{% endif %}>
                                {{ player }}
                            </option>
                        {% endfor %}
                    </select>
                {% endfor %}
            </div>

            <div class="col">
                <h3>{{ team2 }}</h3>

                {% for i in range(11) %}
                    <label>Player {{ i + 1 }}</label>
                    <select name="team2_player_{{ i }}" onchange="this.form.submit()">
                        {% for player in team2_player_list %}
                            <option value="{{ player }}"
                                {% if team2_players|length > i and team2_players[i] == player %}selected{% endif %}>
                                {{ player }}
                            </option>
                        {% endfor %}
                    </select>
                {% endfor %}
            </div>
        </div>

        <h2>📊 Player Insights</h2>

        <div class="row">
            <div class="col">
                <h3>{{ team1 }} Batting</h3>

                {% if team1_batting %}
                <table>
                    <tr>
                        {% for key in team1_batting[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                    </tr>

                    {% for row in team1_batting %}
                    <tr>
                        {% for value in row.values() %}
                            <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>

            <div class="col">
                <h3>{{ team2 }} Batting</h3>

                {% if team2_batting %}
                <table>
                    <tr>
                        {% for key in team2_batting[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                    </tr>

                    {% for row in team2_batting %}
                    <tr>
                        {% for value in row.values() %}
                            <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
        </div>

        <div class="row">
            <div class="col">
                <h3>{{ team1 }} Bowling</h3>

                {% if team1_bowling %}
                <table>
                    <tr>
                        {% for key in team1_bowling[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                    </tr>

                    {% for row in team1_bowling %}
                    <tr>
                        {% for value in row.values() %}
                            <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>

            <div class="col">
                <h3>{{ team2 }} Bowling</h3>

                {% if team2_bowling %}
                <table>
                    <tr>
                        {% for key in team2_bowling[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                    </tr>

                    {% for row in team2_bowling %}
                    <tr>
                        {% for value in row.values() %}
                            <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
        </div>

        <button type="submit" name="predict" value="yes">🏆 Predict Winner</button>
    </form>

    {% if result %}
        <div class="result">
            <h2>🏆 Prediction Result</h2>

            <div class="score-box">
                <div class="score-card">
                    <h3>{{ team1 }}</h3>
                    <div class="score">{{ score1 }}</div>
                    <p>Predicted Score</p>
                </div>

                <div class="score-card">
                    <h3>{{ team2 }}</h3>
                    <div class="score">{{ score2 }}</div>
                    <p>Predicted Score</p>
                </div>
            </div>

            <h2>Winner: {{ winner }}</h2>
            <p><b>Expected Margin:</b> {{ margin }} runs</p>
        </div>

        <h2>🧠 Why this prediction?</h2>
        <ul>
            {% for reason in reasons %}
                <li>{{ reason }}</li>
            {% endfor %}
        </ul>
    {% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    team1 = teams[0]
    team2 = teams[1]

    if request.method == "POST":
        team1 = request.form.get("team1", teams[0])
        team2 = request.form.get("team2", teams[1])

    if team1 == team2:
        team2 = teams[1] if team1 != teams[1] else teams[0]

    team1_player_list = get_team_players(team1)
    team2_player_list = get_team_players(team2)

    team1_players = get_default_players(team1)
    team2_players = get_default_players(team2, exclude=team1_players)

    # Important fix:
    # Player selections update on every POST, not only when Predict button is clicked.
    if request.method == "POST":
        selected1 = [request.form.get(f"team1_player_{i}") for i in range(11)]
        selected2 = [request.form.get(f"team2_player_{i}") for i in range(11)]

        selected1 = [p for p in selected1 if p in team1_player_list]
        selected2 = [p for p in selected2 if p in team2_player_list]

        if len(selected1) == 11:
            team1_players = selected1

        if len(selected2) == 11:
            team2_players = selected2

    h2h = head_to_head_insights(team1, team2)

    error = None
    result = False
    score1 = None
    score2 = None
    winner = None
    margin = None
    reasons = []

    team1_batting = batting_table(team1_players)
    team2_batting = batting_table(team2_players)
    team1_bowling = bowling_table(team1_players)
    team2_bowling = bowling_table(team2_players)

    if request.method == "POST" and request.form.get("predict") == "yes":
        if len(set(team1_players)) != 11:
            error = f"Duplicate players selected in {team1}."

        elif len(set(team2_players)) != 11:
            error = f"Duplicate players selected in {team2}."

        elif set(team1_players).intersection(set(team2_players)):
            common = ", ".join(sorted(set(team1_players).intersection(set(team2_players))))
            error = "Same player cannot be selected in both teams: " + common

        else:
            input1 = calculate_features(team1, team2, team1_players, team2_players)
            input2 = calculate_features(team2, team1, team2_players, team1_players)

            pred1 = model.predict(input1)[0]
            pred2 = model.predict(input2)[0]

            score1 = round(pred1)
            score2 = round(pred2)

            if pred1 > pred2:
                winner = team1
                margin = round(pred1 - pred2)
            elif pred2 > pred1:
                winner = team2
                margin = round(pred2 - pred1)
            else:
                winner = "Close Match"
                margin = 0

            result = True

            team1_sr = bat_stats[bat_stats["batter"].isin(team1_players)]["strike_rate"].mean()
            team2_sr = bat_stats[bat_stats["batter"].isin(team2_players)]["strike_rate"].mean()

            team1_wk = bowl_stats[bowl_stats["bowler"].isin(team1_players)]["wickets"].mean()
            team2_wk = bowl_stats[bowl_stats["bowler"].isin(team2_players)]["wickets"].mean()

            team1_eco = bowl_stats[bowl_stats["bowler"].isin(team1_players)]["economy"].mean()
            team2_eco = bowl_stats[bowl_stats["bowler"].isin(team2_players)]["economy"].mean()

            if team1_sr > team2_sr:
                reasons.append(f"{team1} has stronger batting strike rate.")
            else:
                reasons.append(f"{team2} has stronger batting strike rate.")

            if team1_wk > team2_wk:
                reasons.append(f"{team1} has better wicket-taking bowling strength.")
            else:
                reasons.append(f"{team2} has better wicket-taking bowling strength.")

            if team1_eco < team2_eco:
                reasons.append(f"{team1} bowlers have better economy rate.")
            else:
                reasons.append(f"{team2} bowlers have better economy rate.")

            if h2h["matches"] > 0:
                reasons.append(
                    f"Historically, {team1} won {h2h['team1_wins']} times and {team2} won {h2h['team2_wins']} times against each other."
                )

    return render_template_string(
        HTML,
        teams=teams,
        team1=team1,
        team2=team2,
        h2h=h2h,
        team1_player_list=team1_player_list,
        team2_player_list=team2_player_list,
        team1_players=team1_players,
        team2_players=team2_players,
        error=error,
        result=result,
        score1=score1,
        score2=score2,
        winner=winner,
        margin=margin,
        reasons=reasons,
        team1_batting=team1_batting,
        team2_batting=team2_batting,
        team1_bowling=team1_bowling,
        team2_bowling=team2_bowling
    )


if __name__ == "__main__":
    app.run(debug=True)