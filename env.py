import requests

def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data


league = input("Enter the league you would like to use: ")
year = str(input("Enter the year of the league you would like to use, (e.g. 19/20): "))

SEASON=year
TOURNAMENT_ID=fetch_and_parse_json(f"http://www.sofascore.com/api/v1/search/unique-tournaments?q={league}&page=0")["results"][0]["entity"]["id"]
seasons = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/seasons")['seasons']
for x in seasons:
    if x["year"] == year:
        SEASON_ID = x["id"]
        break
    else:
        SEASON_ID = None


OTHER_TOURNAMENT_ID = fetch_and_parse_json(f"https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/round/1")["events"][0]["tournament"]["id"]