import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def check_website(url):
    try:
        response = requests.get(url)
        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException as e:
        # Handle any exceptions (like network errors)
        print(f"Error checking {url}: {e}")
        return False

def add_website_if_valid(url, website_list):
    if check_website(url):
        website_list.append(url)

def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data

def check_website_and_assign(url):
    try:
        response = requests.get(url)
        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            variable = response.json()
        else:
            variable = None
    except requests.RequestException as e:
        # Handle any exceptions (like network errors)
        print(f"Error checking {url}: {e}")
        variable = None

    return variable


def create_training_testing_data():


    def create_match_dataframe(match_data):
        columns = [
            "Home_GD_1", "Home_GD_2", "Home_GD_3", "Home_GD_4", "Home_GD_5",
            "Home_League_Pos", "League_Round", 'Draw_odds', "Home_Win_Odds", "Home_Points",
            "Away_GD_1", "Away_GD_2", "Away_GD_3", "Away_GD_4", "Away_GD_5",
            "Away_League_Pos", "Away_Win_Odds", "Away_Points"
        ]

        formatted_data = []

        for home_data, away_data in match_data:
            formatted_data.append(home_data + away_data)  # Combine home and away data

        return pd.DataFrame(formatted_data, columns=columns)

    def create_tuple_list():

        def get_odds_from_matchid(matchid):
            odds = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/event/{matchid}/odds/1/featured")['featured']['default']['choices']
            one = 1 / (float(odds[0]['fractionalValue'].split("/")[0]) / float(odds[0]['fractionalValue'].split("/")[1]) + 1)
            two = 1 / (float(odds[1]['fractionalValue'].split("/")[0]) / float(odds[1]['fractionalValue'].split("/")[1]) + 1)
            three = 1 / (float(odds[2]['fractionalValue'].split("/")[0]) / float(odds[2]['fractionalValue'].split("/")[1]) + 1)
            total = one + two + three
            home = round((one / total), 2)
            draw = round((two / total), 2)
            away = round((three / total), 2)
            return home, draw, away


        def leagueid():
            league = input("What league would you like to create training/testing data of? ")
            leagueid = \
                fetch_and_parse_json(
                    "http://www.sofascore.com/api/v1/search/unique-tournaments?q=" + league + "&page=0")[
                    'results'][0]['entity']['id']
            return leagueid

        def seasonid(leagueid):
            options = []
            src = []
            seasons = \
                fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/seasons")[
                    'seasons']
            for t in range(0, len(seasons)):
                add_website_if_valid(
                    "http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(
                        seasons[t]['id']) + "/events/round/1", src)
            for i in range(0, len(src)):
                id = seasons[i]['id']
                random_event_id = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueid}/season/{id}/events/round/10")['events'][0]['id']
                if check_website(f"http://www.sofascore.com/api/v1/event/{random_event_id}/pregame-form") and check_website(f"http://www.sofascore.com/api/v1/event/{random_event_id}/odds/1/featured"):
                    options.append(seasons[i]['year'])
                    print(str(i + 1) + ". " + options[len(options) - 1])
            year = int(input("Which season number would you like to load? "))
            for l in range(0, len(seasons)):
                if seasons[l]['year'] == options[year - 1]:
                    return seasons[l]['id']

        def match_list(seasonid, leagueid, round):
            full_list = []
            for p in range(0, int(round)):
                roundmatches = fetch_and_parse_json(
                    "http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(
                        seasonid) + "/events/round/" + str(p + 1))["events"]
                for i in range(0, len(roundmatches)):
                    match = roundmatches[i]
                    matchid = match["id"]
                    odds = None
                    odds = check_website_and_assign(f"http://www.sofascore.com/api/v1/event/{matchid}/odds/1/featured")
                    if odds != None:
                        full_list.append(matchid)
            return full_list

        # creates information for the training data
        leagueid = leagueid()
        seasonid = seasonid(leagueid)
        id_list = match_list(seasonid, leagueid, 38)
        home_team_list = []
        away_team_list = []

        for i in range(0, len(id_list)):
            print(f"{round(((i+1)/len(id_list))*100, 2)}% complete")
            form = None
            form = check_website_and_assign(f"http://www.sofascore.com/api/v1/event/{id_list[i]}/pregame-form")
            if form is not None:
                if len(form['homeTeam']['form']) > 4 and len(form['awayTeam']['form']) > 4:
                    game = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/event/{id_list[i]}")['event']
                    round_num = game['roundInfo']['round']
                    if len(game['homeScore']) > 0 and len(game['awayScore']) > 0:
                        if game['homeScore']['current'] > game['awayScore']['current']:
                            home_points = 3
                            away_points = 0
                        elif game['homeScore']['current'] < game['awayScore']['current']:
                            home_points = 0
                            away_points = 3
                        else:
                            home_points = 1
                            away_points = 1
                        home_win_odds, draw_odds, away_win_odds = get_odds_from_matchid(id_list[i])
                        home_id = game['homeTeam']['id']
                        home_last_5 = []
                        away_id = game['awayTeam']['id']
                        away_last_5 = []

                        counter = 0
                        switch = True
                        while switch:
                            page = None
                            page = check_website_and_assign(f"http://www.sofascore.com/api/v1/team/{home_id}/events/last/{counter}")
                            if page is not None:
                                page = page['events']
                                page2 = fetch_and_parse_json(
                                    f"http://www.sofascore.com/api/v1/team/{home_id}/events/last/{counter + 1}")['events']
                                for x in range(len(page)-1, -1, -1):
                                    if page[x]['id'] == id_list[i]:
                                        while len(home_last_5) < 5:
                                            for y in range(x-1, -1, -1):
                                                if page[y]['tournament']['uniqueTournament']['id'] == leagueid and len(home_last_5) < 5 and len(page[y]['homeScore']) > 0 and len(page[y]['awayScore']) > 0:
                                                    if page[y]['homeTeam']['id'] == home_id:
                                                        home_last_5.append(page[y]['homeScore']['current'] - page[y]['awayScore']['current'])
                                                    else:
                                                        home_last_5.append(page[y]['awayScore']['current'] - page[y]['homeScore']['current'])
                                            if len(home_last_5) < 5:
                                                for y in range(len(page2)-1, -1, -1):
                                                    if page2[y]['tournament']['uniqueTournament']['id'] == leagueid and len(home_last_5) < 5 and len(page2[y]['homeScore']) > 0 and len(page2[y]['awayScore']) > 0:
                                                        if page2[y]['homeTeam']['id'] == home_id:
                                                            home_last_5.append(page2[y]['homeScore']['current'] - page2[y]['awayScore']['current'])
                                                        else:
                                                            home_last_5.append(page2[y]['awayScore']['current'] - page2[y]['homeScore']['current'])
                            if len(home_last_5) == 5:
                                switch = False
                            counter += 1

                        ## Do the same for the away team
                        counter = 0
                        switch2 = True
                        while switch2:
                            page = None
                            page = check_website_and_assign(f"http://www.sofascore.com/api/v1/team/{away_id}/events/last/{counter}")
                            if page is not None:
                                page = page['events']
                                page2 = fetch_and_parse_json(
                                    f"http://www.sofascore.com/api/v1/team/{away_id}/events/last/{counter + 1}")['events']
                                for x in range(len(page)-1, -1, -1):
                                    if page[x]['id'] == id_list[i]:
                                        while len(away_last_5) < 5:
                                            for y in range(x-1, -1, -1):
                                                if page[y]['tournament']['uniqueTournament']['id'] == leagueid and len(away_last_5) < 5 and len(page[y]['homeScore']) > 0 and len(page[y]['awayScore']) > 0:
                                                    if page[y]['homeTeam']['id'] == away_id:
                                                        away_last_5.append(page[y]['homeScore']['current'] - page[y]['awayScore']['current'])
                                                    else:
                                                        away_last_5.append(page[y]['awayScore']['current'] - page[y]['homeScore']['current'])
                                            if len(away_last_5) < 5:
                                                for y in range(len(page2)-1, -1, -1):
                                                    if page2[y]['tournament']['uniqueTournament']['id'] == leagueid and len(away_last_5) < 5 and len(page2[y]['awayScore']) > 0 and len(page2[y]['homeScore']) > 0:
                                                        if page2[y]['homeTeam']['id'] == away_id:
                                                            away_last_5.append(page2[y]['homeScore']['current'] - page2[y]['awayScore']['current'])
                                                        else:
                                                            away_last_5.append(page2[y]['awayScore']['current'] - page2[y]['homeScore']['current'])
                            if len(away_last_5) == 5:
                                switch2 = False
                            counter += 1

                        if len(home_last_5) == 5 and len(away_last_5) == 5 and home_win_odds > 0 and draw_odds > 0 and away_win_odds > 0:
                            home_team_list.append((home_last_5[0], home_last_5[1], home_last_5[2], home_last_5[3], home_last_5[4], form['homeTeam']['position'], round_num, draw_odds, home_win_odds, home_points))
                            away_team_list.append((away_last_5[0], away_last_5[1], away_last_5[2], away_last_5[3], away_last_5[4], form['awayTeam']['position'], away_win_odds, away_points))
        return home_team_list, away_team_list


    eghome, egaway = create_tuple_list()
    combined = []
    if len(eghome) == len(egaway):
        for c in range(0, len(eghome)):
            combined.append((eghome[c], egaway[c]))


    # Path to the /data folder
    data_folder = 'data'

    dataframe = create_match_dataframe(combined)


    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Ask the user for the action they want to perform
    action = input("Would you like to:\n1. Append to an existing JSON file\n2. Replace an existing JSON file\n3. Create a new JSON file\nPlease select 1, 2, or 3: ")

    # Get the list of JSON files in the /data folder
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    if action == '1' or action == '2':
        if json_files:
            print("\nAvailable JSON files in '/data' folder:")
            for i, file in enumerate(json_files, 1):
                print(f"{i}. {file}")
            file_choice = int(input("\nPlease choose a file to append to/replace (enter the number): ")) - 1

            if 0 <= file_choice < len(json_files):
                chosen_file = os.path.join(data_folder, json_files[file_choice])
                if action == '1':
                    # Append to the chosen file
                    with open(chosen_file, 'r') as f:
                        data = json.load(f)
                    try:
                        new_data = dataframe.to_dict(orient='records')
                        data.extend(new_data)  # Assuming it's a list, adjust accordingly
                        with open(chosen_file, 'w') as f:
                            json.dump(data, f, indent=4)
                        print(f"Data appended to {chosen_file}.")
                    except json.JSONDecodeError:
                        print("Invalid JSON data entered.")
                elif action == '2':
                    # Replace the chosen file
                    try:
                        new_data = dataframe.to_dict(orient='records')
                        with open(chosen_file, 'w') as f:
                            json.dump(new_data, f, indent=4)
                        print(f"File {chosen_file} has been replaced.")
                    except json.JSONDecodeError:
                        print("Invalid JSON data entered.")
            else:
                print("Invalid file choice.")
        else:
            print("No existing JSON files found in the '/data' folder.")
    elif action == '3':
        new_file_name = input("Enter the name for the new JSON file (without the .json extension): ") + ".json"
        new_file_path = os.path.join(data_folder, new_file_name)
        if new_file_name in json_files:
            print("A file with this name already exists.")
        else:
            new_data = dataframe.to_dict(orient='records')
            try:
                with open(new_file_path, 'w') as f:
                    json.dump(new_data, f, indent=4)
                print(f"New file {new_file_name} created in the '/data' folder.")
            except json.JSONDecodeError:
                print("Invalid JSON data entered.")
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

def train_and_evaluate_model(json_file_path):

    def bet_1_on_best_odds_result(predicted_probs, bookies_odds, results, range_ceiling):
        new_pred = []
        new_bookies = []
        new_results = []
        for i in range(len(predicted_probs)):
            if bookies_odds[i][1] < range_ceiling:
                new_pred.append(predicted_probs[i])
                new_bookies.append(bookies_odds[i])
                new_results.append(results[i])
        predicted = [((round(1/home, 2), "W"), (round(1/draw, 2), "D"), (round(1/away, 2), "L")) for home, draw, away in new_pred]
        predicted = [min(inner_tuple, key=lambda x: x[0]) for inner_tuple in predicted]
        bookies = [((round(1/home, 2), "W"), (round(1/draw, 2), "D"), (round(1/away, 2), "L")) for home, draw, away in new_bookies]
        total = 0
        for i in range(len(predicted)):
            if predicted[i][1] == new_results[i]:
                for x in bookies[i]:
                    if x[1] == predicted[i][1]:
                        total += (x[0]-1)*(2/predicted[i][0])
            else:
                total -= (2/predicted[i][0])
        return total

    # Load data from JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Define target variable (Home_Points) and features (X)
    y = df["Home_Points"].astype(int)  # Ensure it's categorical
    X = df.drop(columns=["Home_Points", "Away_Points"])  # Features

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )
    y_test_labels = ["W" if points == 3 else "D" if points == 1 else "L" for points in y_test]

    # Extract bookmaker probabilities for the test set
    bookie_probs = list(zip(X_test_df["Home_Win_Odds"],
                            X_test_df["Draw_odds"],
                            X_test_df["Away_Win_Odds"]))

    # Now scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Train a classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)

    # Ensure the order of classes matches expected output (home win, draw, away win)
    class_order = model.classes_  # Get the order of classes used by the model
    prob_tuples = [tuple(probs[class_order == c][0] for c in [3, 1, 0]) for probs in y_pred_probs]
    prob_tuples = [(float(home), float(draw), float(away)) for home, draw, away in prob_tuples]


    ceiling = 0.20
    print(f"Profit based on model confidence: {bet_1_on_best_odds_result(prob_tuples, bookie_probs, y_test_labels, ceiling)}")
    print(f"Profit based on bookies confidence: {bet_1_on_best_odds_result(bookie_probs, bookie_probs, y_test_labels, ceiling)}")

    # Predict classes
    y_pred_mapped = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_mapped):.4f}")
    # Generate and display a confusion matrix
    cm = confusion_matrix(y_test, y_pred_mapped, labels=[0, 1, 3])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 3], yticklabels=[0, 1, 3])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return model, scaler  # Return model & scaler for future predictions

# Enter training and testing data file paths to predict a final league table

# Run this function to create training and testing data for the second predictive model
#create_training_testing_data()

# Train and evaluate the second predictive model based on the generated data
model, scaler = train_and_evaluate_model("data/Prem With Odds.json")

