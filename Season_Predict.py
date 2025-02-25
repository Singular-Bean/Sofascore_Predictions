import requests
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data


# Set Pandas option to display all columns
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Adjust the width to prevent wrapping


def create_training_or_testing_data():
    league = input("Enter the league you would like to use: ")
    year = str(input("Enter the year of the league you would like to use, (e.g. 19/20): "))

    TOURNAMENT_ID = \
    fetch_and_parse_json(f"http://www.sofascore.com/api/v1/search/unique-tournaments?q={league}&page=0")["results"][0][
        "entity"]["id"]
    seasons = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/seasons")[
        'seasons']
    for x in seasons:
        if x["year"] == year:
            SEASON_ID = x["id"]
            break
        else:
            SEASON_ID = None

    OTHER_TOURNAMENT_ID = fetch_and_parse_json(
        f"https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/round/1")[
        "events"][0]["tournament"]["id"]

    TARGET_FILENAME = input("Enter the file path you would like to save the data to: ")

    def teamidlist(tournamentid, seasonid):
        teamlist = []
        url = "http://www.sofascore.com/api/v1/unique-tournament/" + str(tournamentid) + "/season/" + str(
            seasonid) + "/standings/total"
        data = fetch_and_parse_json(url)['standings'][0]['rows']
        for i in range(0, len(data)):
            teamlist.append(data[i]['team']['id'])
        return teamlist

    teamlistids = teamidlist(TOURNAMENT_ID, SEASON_ID)
    number_of_teams_in_comp = len(teamlistids)

    def match_result_list(seasonid, tournamentid):
        full_list = []
        not_completed_list = []
        for p in range(0, number_of_teams_in_comp - 1):
            roundmatches = \
            fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(tournamentid) +
                                 "/season/" + str(seasonid) + "/events/round/" + str(p + 1))["events"]
            for i in range(0, len(roundmatches)):
                match = roundmatches[i]
                if match["status"]["type"] == "finished":
                    full_list.append(
                        (match["homeTeam"]["name"], match["homeScore"]["current"], match["awayTeam"]["name"],
                         match["awayScore"]["current"]))
                elif match["status"]["type"] == "notstarted" and (
                match["homeTeam"]["name"], match["awayTeam"]["name"]) not in not_completed_list:
                    not_completed_list.append((match["homeTeam"]["name"], match["awayTeam"]["name"]))
        return full_list, not_completed_list

    team_and_final_position = []
    for i in range(0, len(teamlistids)):
        link = fetch_and_parse_json(
            f"https://www.sofascore.com/api/v1/tournament/{OTHER_TOURNAMENT_ID}/season/{SEASON_ID}/standings/total")[
            "standings"][0]["rows"][i]
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
            score = input(
                "Would you like to enter the scores for these matches?:\n" + f"{home} vs {away}" + ": ").split("-")
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


def season_predict(training_data, testing_data):
    def thin_down_list(input_list):
        result = []
        seen_teams = set()
        seen_positions = set()

        for chance, team_name, position in input_list:
            # Check if the team or position has already been seen
            if team_name not in seen_teams and position not in seen_positions:
                result.append((chance, team_name, position))
                seen_teams.add(team_name)
                seen_positions.add(position)

        return result

    # Load training data
    with open(training_data, "r") as file:
        training_data = json.load(file)

    # Load testing data
    with open(testing_data, "r") as file:
        testing_data = json.load(file)

    # Convert JSON to DataFrame
    train_df = pd.DataFrame(training_data)
    test_df = pd.DataFrame(testing_data)

    # Prepare features and target
    X_train = train_df.drop("Final Position", axis=1).drop("Team", axis=1)
    y_train = train_df["Final Position"]

    X_test = test_df.drop("Final Position", axis=1).drop("Team", axis=1)
    y_test = test_df["Final Position"]
    names = test_df["Team"]

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict models
    y_proba = model.predict_proba(X_test).tolist()
    y_pred = model.predict(X_test)

    # Example probabilities matrix (rows: teams, columns: positions)
    # Each value represents P(team finishes in that position)
    probabilities = np.array(y_proba)

    # Convert to a cost matrix (negative probabilities because we're maximizing)
    cost_matrix = -probabilities

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Output the result
    assigned = [(row, col, probabilities[row, col]) for row, col in zip(row_indices, col_indices)]

    # Print assignments
    assigned_positions = []
    for team_idx, position_idx, prob in assigned:
        assigned_positions.append((names[team_idx], position_idx + 1, prob))

    # Display resolved positions
    resolved_table = pd.DataFrame(
        assigned_positions, columns=["Team", "Assigned Position", "Probability"]
    ).sort_values(by="Assigned Position")
    print(resolved_table)

    # Extract the resolved positions for the scatter plot
    resolved_positions = [pos for _, pos, _ in assigned_positions]

    # Calculate and display model accuracy
    accuracy = accuracy_score(y_test, resolved_positions)
    print(f"Model accuracy: {accuracy:.2f}")

    # Correlation coefficient
    print("Correlation Coefficient:", np.corrcoef(y_test, resolved_positions)[0, 1])

    # Plot the results
    plt.scatter(resolved_positions, y_test)
    plt.xlabel("Predicted Position")
    plt.ylabel("Actual Position")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # Update ticks
    xticks = range(len(assigned), 0, -1)
    yticks = range(len(assigned), 0, -1)
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(yticks)

    # Show the plot
    plt.show()


option = input(
    "What would you like to do:\n1. Create training/testing Data\n2. Train a model to predict a season\nPlease select 1 or 2: ")
if option == "1":
    create_training_or_testing_data()
elif option == "2":
    chosen_training_data = input("Enter the file path of the training data (e.g: data/training_data.json): ")
    chosen_testing_data = input("Enter the file path of the testing data (e.g: data/testing_data.json): ")
    season_predict(chosen_training_data, chosen_testing_data)
