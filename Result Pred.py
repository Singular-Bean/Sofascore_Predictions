import requests


def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data


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


def get_odds_from_matchid(matchid):
    odds = \
    fetch_and_parse_json(f"http://www.sofascore.com/api/v1/event/{matchid}/odds/1/featured")['featured']['default'][
        'choices']
    one = 1 / (float(odds[0]['fractionalValue'].split("/")[0]) / float(odds[0]['fractionalValue'].split("/")[1]) + 1)
    two = 1 / (float(odds[1]['fractionalValue'].split("/")[0]) / float(odds[1]['fractionalValue'].split("/")[1]) + 1)
    three = 1 / (float(odds[2]['fractionalValue'].split("/")[0]) / float(odds[2]['fractionalValue'].split("/")[1]) + 1)
    total = one + two + three
    home = round((one / total), 2)
    draw = round((two / total), 2)
    away = round((three / total), 2)
    return home, draw, away




id_list = [12436870, 12436871, 12436872, 12436873, 12436874, 12436875, 12436877, 12436879, 12436880, 12436881, 12436888,
           12436892, 12436894, 12436898, 12436900, 12436902, 12436886, 12436884, 12436904, 12436896, 12436906, 12436908,
           12436912, 12436915, 12436918, 12436926, 12436927, 12436910, 12436923, 12436920, 12436949, 12436936, 12436938,
           12436941, 12436944, 12436946, 12436933, 12436931, 12436951, 12436954, 12436980, 12436957, 12436964, 12436967,
           12436969, 12436973, 12436977, 12436962, 12436959, 12436971, 12436999, 12436985, 12436987, 12436989, 12436992,
           12437001, 12437002, 12436993, 12436995, 12437015, 12437004, 12437008, 12437020, 12437022, 12437024, 12437018,
           12437006, 12437013, 12437011, 12437038, 12437028, 12437034, 12437035, 12437037, 12437030, 12437026, 12437039,
           12437032, 12437036, 12437048, 12437041, 12437043, 12437044, 12437049, 12437047, 12437045, 12437046, 12437050,
           12437040, 12437056, 12437051, 12437053, 12437054, 12437057, 12437058, 12437060, 12437059, 12437055, 12437052,
           12436876, 12436483, 12436486, 12436485, 12436882, 12436484, 12436494, 12436487, 12436492, 12436890, 12436495,
           12436489, 12436493, 12436491, 12436488, 12436490, 12436499, 12436506, 12436497, 12436496, 12436501, 12436500,
           12436502, 12436503, 12436922, 12436914, 12436498, 12436505, 12436507, 12436504, 12436961, 12436966, 12436947,
           12436972, 12436978, 12436984, 12436935, 12436942, 12436955, 13015216, 12436990, 12436450, 12436445, 12436449,
           12436459, 12436462, 12436444, 12436441, 12436454, 13344418, 12436438, 12436446, 12436460, 12436456, 12436447,
           12436457, 12436451, 12436461, 12436458, 12436453, 12436463, 12436467, 12436478, 12436464, 12436466, 12436468,
           12436469, 12436470, 12436471, 12436481, 12436476, 12436473, 12436477, 12436479, 12436480, 12436482, 12436474,
           12436475, 12436472, 12436512, 12436517, 12436515, 12436521, 12436513, 12436530, 12436510, 12436522, 12436410,
           12436519, 12436523, 12436518, 12436419, 12436508, 12436520, 12436528, 12436525, 12436514, 12436437, 12436436,
           12436531, 12436509, 12436526, 12436434, 12436527, 12436529, 12436511, 12436524, 12436516, 12436885, 12436440,
           12436442, 12436889, 12436443, 12436448, 12436883, 12436439, 12436455, 12436891, 12436895, 12436901, 12436905,
           12436909, 12436903, 12436897, 12436907, 12436893, 12436899, 12436930, 12436911, 12436921, 12436924, 12436928,
           12436932, 12436916, 12436925, 12436913, 12436919, 12436937, 12436943, 12436934, 12436940, 12436948, 12436956,
           12436939, 12436945, 12436952, 12437019]
leagueid = 17
home_team_list = []
away_team_list = []

for i in range(0, len(id_list)):
    print(f"{round(((i + 1) / len(id_list)) * 100, 2)}% complete")
    if check_website(f"http://www.sofascore.com/api/v1/event/{id_list[i]}/pregame-form"):
        form = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/event/{id_list[i]}/pregame-form")
        if len(form['homeTeam']['form']) > 4 and len(form['awayTeam']['form']) > 4:
            game = fetch_and_parse_json(f"http://www.sofascore.com/api/v1/event/{id_list[i]}")['event']
            round_num = game['roundInfo']['round']
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
                if check_website(f"http://www.sofascore.com/api/v1/team/{home_id}/events/last/{counter}"):
                    page = \
                    fetch_and_parse_json(f"http://www.sofascore.com/api/v1/team/{home_id}/events/last/{counter}")[
                        'events']
                    for x in range(len(page) - 1, -1, -1):
                        if page[x]['id'] == id_list[i]:
                            while len(home_last_5) < 5:
                                for y in range(x - 1, -1, -1):
                                    if page[y]['tournament']['uniqueTournament']['id'] == leagueid:
                                        if page[y]['homeTeam']['id'] == home_id and len(home_last_5) < 5 and len(
                                                page[y]['homeScore']) > 0 and len(page[y]['awayScore']) > 0:
                                            home_last_5.append(
                                                page[y]['homeScore']['current'] - page[y]['awayScore']['current'])
                                        elif len(home_last_5) < 5 and len(page[y]['awayScore']) > 0 and len(
                                                page[y]['homeScore']) > 0:
                                            #print(page[y]['homeScore'])
                                            home_last_5.append(
                                                page[y]['awayScore']['current'] - page[y]['homeScore']['current'])
                                page2 = fetch_and_parse_json(
                                    f"http://www.sofascore.com/api/v1/team/{home_id}/events/last/{counter + 1}")[
                                    'events']
                                if len(home_last_5) < 5:
                                    for y in range(len(page2) - 1, -1, -1):
                                        if page2[y]['tournament']['uniqueTournament']['id'] == leagueid:
                                            if page2[y]['homeTeam']['id'] == home_id and len(home_last_5) < 5 and len(
                                                    page2[y]['homeScore']) > 0 and len(page2[y]['awayScore']) > 0:
                                                home_last_5.append(
                                                    page2[y]['homeScore']['current'] - page2[y]['awayScore']['current'])
                                            elif page2[y]['awayTeam']['id'] == home_id and len(home_last_5) < 5 and len(
                                                    page2[y]['awayScore']) > 0 and len(page2[y]['homeScore']) > 0:
                                                home_last_5.append(
                                                    page2[y]['awayScore']['current'] - page2[y]['homeScore']['current'])
                if len(home_last_5) == 5:
                    switch = False
                counter += 1

                ## Do the same for the away team
            counter = 0
            switch2 = True
            while switch2:
                if check_website(f"http://www.sofascore.com/api/v1/team/{away_id}/events/last/{counter}"):
                    page = \
                    fetch_and_parse_json(f"http://www.sofascore.com/api/v1/team/{away_id}/events/last/{counter}")[
                        'events']
                    for x in range(len(page) - 1, -1, -1):
                        if page[x]['id'] == id_list[i]:
                            #print(id_list[i])
                            while len(away_last_5) < 5:
                                for y in range(x - 1, -1, -1):
                                    if page[y]['tournament']['uniqueTournament']['id'] == leagueid and len(
                                            page[y]['homeScore']) > 0 and len(page[y]['awayScore']) > 0:
                                        if page[y]['homeTeam']['id'] == away_id and len(away_last_5) < 5 and len(
                                                page[y]['awayScore']) > 0 and len(page[y]['homeScore']) > 0:
                                            #print(page[y]['homeScore'])
                                            #print(id_list[i])
                                            away_last_5.append(
                                                page[y]['homeScore']['current'] - page[y]['awayScore']['current'])
                                        elif len(away_last_5) < 5 and len(page[y]['awayScore']) > 0 and len(
                                                page[y]['homeScore']) > 0:
                                            away_last_5.append(
                                                page[y]['awayScore']['current'] - page[y]['homeScore']['current'])
                                page2 = fetch_and_parse_json(
                                    f"http://www.sofascore.com/api/v1/team/{away_id}/events/last/{counter + 1}")[
                                    'events']
                                if len(away_last_5) < 5:
                                    for y in range(len(page2) - 1, -1, -1):
                                        if page2[y]['tournament']['uniqueTournament']['id'] == leagueid:
                                            if page2[y]['homeTeam']['id'] == away_id and len(away_last_5) < 5 and len(
                                                    page2[y]['awayScore']) > 0 and len(page2[y]['homeScore']) > 0:
                                                away_last_5.append(
                                                    page2[y]['homeScore']['current'] - page2[y]['awayScore']['current'])
                                            elif page2[y]['awayTeam']['id'] == away_id and len(away_last_5) < 5 and len(
                                                    page2[y]['awayScore']) > 0 and len(page2[y]['homeScore']) > 0:
                                                away_last_5.append(
                                                    page2[y]['awayScore']['current'] - page2[y]['homeScore']['current'])
                if len(away_last_5) == 5:
                    switch2 = False
                counter += 1

            if len(home_last_5) == 5 and len(
                    away_last_5) == 5 and home_win_odds > 0 and draw_odds > 0 and away_win_odds > 0:
                print(home_win_odds, draw_odds, away_win_odds, id_list[i])
                home_team_list.append((home_last_5[0], home_last_5[1], home_last_5[2], home_last_5[3], home_last_5[4],
                                       form['homeTeam']['position'], round_num, draw_odds, home_win_odds, home_points))
                away_team_list.append((away_last_5[0], away_last_5[1], away_last_5[2], away_last_5[3], away_last_5[4],
                                       form['awayTeam']['position'], round_num, away_win_odds, away_points))
