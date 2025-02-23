import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

season_predict("data/champmidseason-tables-training.json", "data/champmidseason-tables-testing.json")
