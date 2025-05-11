import json
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import seaborn as sns


def create_json_file():
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
        response.raise_for_status()  # Ensure we raise an error for bad status codes
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

    def leagueid():
        league = input("What league would you like to create training/testing data of? ")
        leagueid = \
            fetch_and_parse_json("http://www.sofascore.com/api/v1/search/unique-tournaments?q=" + league + "&page=0")[
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
            random_event_id = fetch_and_parse_json(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueid}/season/{id}/events/round/1")['events'][
                0][
                'id']
            event_lineups = check_website_and_assign(
                f"https://www.sofascore.com/api/v1/event/{random_event_id}/lineups")
            if event_lineups is not None:
                first_player = event_lineups['home']['players'][0]
                if 'statistics' in first_player:
                    options.append(seasons[i]['year'])
                    print(str(i + 1) + ". " + options[len(options) - 1])
        year = int(input("Which season number would you like to load? "))
        for l in range(0, len(seasons)):
            if seasons[l]['year'] == options[year - 1]:
                return [seasons[l]['id'], seasons[l]['year']]

    def match_list(seasonid, leagueid, round):
        full_list = []
        for p in range(0, int(round)):
            roundmatches = fetch_and_parse_json(
                "http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(
                    seasonid) + "/events/round/" + str(p + 1))["events"]
            for i in range(0, len(roundmatches)):
                match = roundmatches[i]
                matchid = match["id"]
                if match["status"]["code"] == 100:
                    full_list.append(matchid)
        return full_list

    def get_outfield_player_stats(match_id):
        item = []
        event_lineups = check_website_and_assign(f"http://www.sofascore.com/api/v1/event/{match_id}/lineups")
        home_lineups = event_lineups['home']['players']
        away_lineups = event_lineups['away']['players']
        for i in home_lineups:
            if 'statistics' in i and i['position'] != 'G':
                item.append(i['statistics'])
        for i in away_lineups:
            if 'statistics' in i and i['position'] != 'G':
                item.append(i['statistics'])
        return item

    def get_gk_player_stats(match_id):
        item = []
        event_lineups = check_website_and_assign(f"http://www.sofascore.com/api/v1/event/{match_id}/lineups")
        home_lineups = event_lineups['home']['players']
        away_lineups = event_lineups['away']['players']
        for i in home_lineups:
            if 'statistics' in i and i['position'] == 'G':
                item.append(i['statistics'])
        for i in away_lineups:
            if 'statistics' in i and i['position'] == 'G':
                item.append(i['statistics'])
        return item

    def initialize_and_update(keys, data_dict):
        # Step 1: Create dictionary with all values set to 0
        result = {key: 0 for key in keys}
        del result['ratingVersions']
        # Step 2: Update values if key exists in data_dict
        for key in data_dict:
            if key in result:
                result[key] = data_dict[key]

        return result

    # creates information for the training data
    leagueid = leagueid()
    uniqueseason = seasonid(leagueid)
    seasonid = uniqueseason[0]
    year = uniqueseason[1].replace("/", "-")
    competitors = len(fetch_and_parse_json(
        f"http://www.sofascore.com/api/v1/unique-tournament/{leagueid}/season/{seasonid}/statistics/info")['teams'])
    totalrounds = (competitors - 1) * 2
    id_list = match_list(seasonid, leagueid, totalrounds)

    choice = input("Would you like to create training/testing data for:\n1. Goalkeepers?\n2. Outfield players? ")
    if choice == '1':
        filepath = f"data/{year}_gk_data.json"
        if os.path.exists(filepath):
            # Read the current contents of the JSON file
            with open(filepath, "r") as f:
                existing_data = json.load(f)
        else:
            # If the file doesn't exist, initialize an empty list
            existing_data = []

        big_list_gk = []
        stat_categories_gk = []
        for x in id_list:
            for y in get_gk_player_stats(x):
                if len(y) > 0:
                    big_list_gk.append(y)
        for w in big_list_gk:
            for z in w:
                if z not in stat_categories_gk:
                    stat_categories_gk.append(z)
        for b in range(0, len(big_list_gk)):
            number = round(len(big_list_gk), -2) / 100
            result_dict = initialize_and_update(stat_categories_gk, big_list_gk[b])
            existing_data.append(result_dict)
            if b % number == 0:
                print(f"{b / number}% of players processed")

    if choice == '2':
        filepath = f"data/{year}_outfield_data.json"
        if os.path.exists(filepath):
            # Read the current contents of the JSON file
            with open(filepath, "r") as f:
                existing_data = json.load(f)
        else:
            # If the file doesn't exist, initialize an empty list
            existing_data = []
        big_list_outfield = []
        stat_categories_outfield = []
        for x in id_list:
            for y in get_outfield_player_stats(x):
                if len(y) > 0:
                    big_list_outfield.append(y)
        for w in big_list_outfield:
            for z in w:
                if z not in stat_categories_outfield:
                    stat_categories_outfield.append(z)
        for b in range(0, len(big_list_outfield)):
            number = round(len(big_list_outfield), -2) / 100
            result_dict = initialize_and_update(stat_categories_outfield, big_list_outfield[b])
            if result_dict['minutesPlayed'] > 7:
                result_dict['moreThan8'] = 1
            else:
                result_dict['moreThan8'] = 0
            existing_data.append(result_dict)
            if b % number == 0:
                print(f"{b / number}% of players processed")

    with open(filepath, "w") as f:
        json.dump(existing_data, f, indent=4)


def train_and_predict(json_file_path):
    json_file_path_eg = "data/23-24_outfield_data.json"
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Define target (y) and features (X)
    y = df["rating"]  # Continuous variable
    X = df.drop(columns=["rating", "moreThan8"])

    # Convert categorical columns if needed
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Apply MinMaxScaler for most models
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    # Apply StandardScaler for SVR
    standard_scaler = StandardScaler()
    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    # Initialize models
    models = {"Random Forest": RandomForestRegressor(n_estimators=100, random_state=42), "OLS Regression": None,
              # Handled separately with statsmodels
              "KNN Regression": KNeighborsRegressor(n_neighbors=5),
              "Decision Tree": DecisionTreeRegressor(random_state=42),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
              "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
              "SVR (RBF Kernel)": SVR(kernel="rbf", C=100, gamma=0.1)}

    # Dictionary to store predictions
    predictions = {}

    # Train & predict with each model
    for name, model in models.items():
        if name == "OLS Regression":
            X_train_ols = sm.add_constant(X_train)
            X_test_ols = sm.add_constant(X_test)
            ols_model = sm.OLS(y_train, X_train_ols).fit()
            predictions[name] = ols_model.predict(X_test_ols)
        elif name == "SVR (RBF Kernel)":
            model.fit(X_train_standard, y_train)  # Use StandardScaler for SVR
            predictions[name] = model.predict(X_test_standard)
        else:
            model.fit(X_train_minmax, y_train)  # Use MinMaxScaler for others
            predictions[name] = model.predict(X_test_minmax)

    # Evaluate models
    for name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MAE = {mae:.3f}, R² = {r2:.3f}")

        # Create a separate scatter plot for each model
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label=f"{name} Predictions", color="blue")
        plt.plot([0, 10], [0, 10], linestyle="--", color="black")  # Perfect prediction line
        plt.xlabel("Actual Ratings")
        plt.ylabel("Predicted Ratings")
        plt.title(f"Model: {name}")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.histplot(y_pred, kde=True, color="green", binwidth=0.5, stat="density")
        plt.axvline(x=y_test.mean(), color="red", linestyle="--", label="Actual Mean")
        plt.title(f"Prediction Distribution for {name}")
        plt.xlabel("Predicted Ratings")
        plt.ylabel("Density")
        plt.legend()
        plt.xlim(0, 10)
        plt.ylim(0.0, 1.0)

        plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(y_test, kde=True, color="green", binwidth=0.5, stat="density")
    plt.axvline(x=y_test.mean(), color="red", linestyle="--", label="Actual Mean")
    plt.title(f"Actual Distribution")
    plt.xlabel("Predicted Ratings")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0.0, 1.0)
    plt.show()


def train_and_use(json_file_path):
    def use_model(model, Xcolumns, scaler):
        testing_data = {}

        for i in Xcolumns:
            testing_data[i] = input(f"What is your subject's {i} statisitc? ")
        df = pd.DataFrame([testing_data])
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df
        prediction = model.predict(df_scaled)
        print(f"Predicted rating: {prediction[0]:.2f}")

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    # Define target (y) and features (X)
    y = df["rating"]  # Continuous variable
    X = df.drop(columns=["rating", "moreThan8"])
    Xcolumns = X.columns.tolist()
    # Convert categorical columns if needed
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Apply MinMaxScaler for most models
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)

    # Apply StandardScaler for SVR
    standard_scaler = StandardScaler()
    X_train_standard = standard_scaler.fit_transform(X_train)

    choice = input(
        "Which model would you like to use for your prediction?\n1. Random Forest\n2. OLS Regression\n3. KNN Regression\n4. Decision Tree\n5. Gradient Boosting\n6. XGBoost\n7. SVR (RBF Kernel)\n")

    if choice == '1':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_minmax, y_train)
        use_model(model, Xcolumns, minmax_scaler)
    elif choice == '2':
        X_train_ols = sm.add_constant(X_train)
        ols_model = sm.OLS(y_train, X_train_ols).fit()
        use_model(ols_model, Xcolumns, None)
    elif choice == '3':
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train_minmax, y_train)
        use_model(model, Xcolumns, minmax_scaler)
    elif choice == '4':
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train_minmax, y_train)
        use_model(model, Xcolumns, minmax_scaler)
    elif choice == '5':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_minmax, y_train)
        use_model(model, Xcolumns, minmax_scaler)
    elif choice == '6':
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_minmax, y_train)
        use_model(model, Xcolumns, minmax_scaler)
    else:
        model = SVR(kernel="rbf", C=100, gamma=0.1)
        model.fit(X_train_standard, y_train)
        use_model(model, Xcolumns, standard_scaler)


options = input(
    "Would you like to:\n1. Create a new JSON file?\n2. Train and test a predictive model?\n3. Train and use a model?\n")

if options == '1':
    create_json_file()
elif options == '2':
    json_file_path = input("Enter the path to the JSON file (e.g. data/24-25_outfield_data.json): ")
    train_and_predict(json_file_path)
elif options == '3':
    json_file_path = input("Enter the path to the JSON file (e.g. data/24-25_outfield_data.json): ")
    train_and_use(json_file_path)
else:
    print("Invalid option. Please choose 1, 2, or 3.")