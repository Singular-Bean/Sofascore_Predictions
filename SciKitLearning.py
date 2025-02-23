import json
import requests
import pandas as pd


# Set Pandas option to display all columns
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Adjust the width to prevent wrapping

# You need to get the tournament and season ids from the sofascore website and put them in env.py
from env import TOURNAMENT_ID, SEASON_ID, SEASON, OTHER_TOURNAMENT_ID

#TARGET_FILENAME = f"data/premmidseason-tables-training.json"
#TARGET_FILENAME = f"data/premmidseason-tables-testing.json"
#TARGET_FILENAME = f"data/premmidseason-tables-testing-24-25-LEWin.json"
#TARGET_FILENAME = f"data/premmidseason-tables-testing-24-25-LEDraw.json"
#TARGET_FILENAME = f"data/premmidseason-tables-testing-24-25-LELoss.json"
#TARGET_FILENAME = f"data/champmidseason-tables-training.json"
#TARGET_FILENAME = f"data/champmidseason-tables-testing.json"
TARGET_FILENAME = f"data/champmidseason-tables-testing24-25.json"
number_of_teams_in_comp = 24



# Clear the contents of the file
#with open(TARGET_FILENAME, "w") as file:
#    pass  # Do nothing, which effectively clears the file


def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data

def teamidlist(tournamentid, seasonid):
    teamlist = []
    url = "http://www.sofascore.com/api/v1/unique-tournament/" + str(tournamentid) + "/season/" + str(seasonid) + "/standings/total"
    data = fetch_and_parse_json(url)['standings'][0]['rows']
    for i in range(0, len(data)):
        teamlist.append(data[i]['team']['id'])
    return teamlist

teamlistids = teamidlist(TOURNAMENT_ID, SEASON_ID)


def match_result_list(seasonid, tournamentid):
    full_list = []
    not_completed_list = []
    for p in range(0, number_of_teams_in_comp-1):
        roundmatches = fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(tournamentid) +
                                            "/season/" + str(seasonid) + "/events/round/" + str(p+1))["events"]
        for i in range(0, len(roundmatches)):
            match = roundmatches[i]
            if match["status"]["type"] == "finished":
                full_list.append((match["homeTeam"]["name"], match["homeScore"]["current"], match["awayTeam"]["name"],
                                match["awayScore"]["current"]))
            elif match["status"]["type"] == "notstarted" and (match["homeTeam"]["name"], match["awayTeam"]["name"]) not in not_completed_list:
                not_completed_list.append((match["homeTeam"]["name"], match["awayTeam"]["name"]))
    return full_list, not_completed_list

team_and_final_position = []
for i in range(0, len(teamlistids)):
    link = fetch_and_parse_json(f"https://www.sofascore.com/api/v1/tournament/{OTHER_TOURNAMENT_ID}/season/{SEASON_ID}/standings/total")["standings"][0]["rows"][i]
    team_and_final_position.append((link["team"]["name"], link["position"]))

position_dict = dict(team_and_final_position)




def create_league_table(results):
    # Dictionary to store team statistics
    teams = {}

    # Process each match result
    for home_team, home_score, away_team, away_score in results:
        # Initialize teams in dictionary if not present
        for team in [home_team, away_team]:
            if team not in teams:
                teams[team] = {"Played": 0, "G/F": 0, "G/A": 0,
                               "G/D": 0, "Pts": 0}

        # Update statistics for home team
        teams[home_team]["Played"] += 1
        teams[home_team]["G/F"] += home_score
        teams[home_team]["G/A"] += away_score
        teams[home_team]["G/D"] = (
            teams[home_team]["G/F"] - teams[home_team]["G/A"]
        )
        if home_score > away_score:  # Home team wins
            teams[home_team]["Pts"] += 3
        elif home_score == away_score:  # Draw
            teams[home_team]["Pts"] += 1

        # Update statistics for away team
        teams[away_team]["Played"] += 1
        teams[away_team]["G/F"] += away_score
        teams[away_team]["G/A"] += home_score
        teams[away_team]["G/D"] = (
            teams[away_team]["G/F"] - teams[away_team]["G/A"]
        )
        if away_score > home_score:  # Away team wins
            teams[away_team]["Pts"] += 3
        elif away_score == home_score:  # Draw
            teams[away_team]["Pts"] += 1

    # Convert the dictionary into a DataFrame
    table = pd.DataFrame(teams).T.reset_index()
    table.rename(columns={"index": "Team"}, inplace=True)

    # Sort by Points, Goal Difference, then Goals For (descending order)
    table = table.sort_values(
        by=["Pts", "G/D", "G/F"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Add Position column
    table.insert(0, "Position", range(1, len(table) + 1))

    # Calculate points difference with the first and last place
    first_place_points = table.loc[0, "Pts"]
    last_place_points = table.loc[len(table) - 1, "Pts"]
    table["Points From First"] = first_place_points - table["Pts"]
    table["Points From Last"] = table["Pts"] - last_place_points


    return table

results, not_completed = match_result_list(SEASON_ID, TOURNAMENT_ID)

if len(not_completed) > 0:
    scores = []
    print("The following matches have not been completed:")
    for home, away in not_completed:
        score = input("Would you like to enter the scores for these matches?:\n"+f"{home} vs {away}"+": ").split("-")
        scores.append((home, int(score[0]), away, int(score[1])))
    results += scores

dataframe = create_league_table(results)

# Step 2: Map the final positions to the "Team" column
dataframe["Final Position"] = dataframe["Team"].map(position_dict)

print(dataframe)
# Write DataFrame to JSON
try:
    with open(TARGET_FILENAME, "r") as file:
        # Load existing JSON data as a list
        data = json.load(file)
except FileNotFoundError:
    with open(TARGET_FILENAME, "w") as file:
        data = []
        pass
data.extend(dataframe.to_dict(orient="records"))

with open(TARGET_FILENAME, "w") as file:
    json.dump(data, file, indent=4)