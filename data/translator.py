import pandas as pd
from math import isnan
import distance
import pickle

valid_player_names = {}

events = pd.read_csv("events.csv")
ginf = pd.read_csv('ginf.csv')
lineups = pd.read_csv("lineups.csv")
labels = ["date", "home_team", "away_team"]
lineups = lineups.drop_duplicates(subset=labels, keep='first')


# budowanie słownika z nazwami piłkarzy
def add_players_to_set(ev, ev_set, op_set):
    pl1, pl2, pl_in, pl_out = ev['player'], ev['player2'], ev['player_in'], ev['player_out']

    if ev['event_type'] == 7:  # substitution
        ev_set.update([pl_in, pl_out])
    # Assumption: every player listed as player2 is from the event team, (except for corners and own goals)
    else:
        is_not_switched = (ev["event_type"] != 2) and (ev["event_type2"] != 15)
        try:
            if isnan(pl1): pass
        except TypeError:  # it is string, therefore it is a name
            if is_not_switched:
                ev_set.add(pl1)
            else:
                op_set.add(pl1)
        try:
            if isnan(pl2): pass
        except TypeError:
            if is_not_switched:
                ev_set.add(pl2)
            else:
                op_set.add(pl2)


def filter_out_correct(player_set, lineup, team, season):
    unmatched = set()
    for pl in player_set:
        if valid_player_names.get((pl, team, season), pl) not in lineup:
            unmatched.add(pl)
    return unmatched


def append_to_dictionary(d, k, e):
    if d.get(k) is None:
        d[k] = [e]
    else:
        d[k].append(e)


def check_fit(full_name, to_fit):
    for word in to_fit:
        if word not in full_name:
            return False
    return True


def resolve_match(pl, lineup, other_lineup, team, other_team, season, match_id,
                  set_to_lineup, lineup_to_set, dists):
    # by word_match
    try:
        if len(set_to_lineup[pl]) == 1:
            potential_name = set_to_lineup[pl][0]
            if len(lineup_to_set[potential_name]) == 1:
                print(f"FOUND PAIRING BY WORD MATCH: {pl} -> {potential_name}")
                valid_player_names[(pl, team, season)] = potential_name
                return True
    except KeyError:
        pass

    # by distance
    (best_score, best_cand), (second_score, _) = dists[0], dists[1]

    if best_score < 0.2 or second_score - best_score > 0.2:
        print(f"FOUND PAIRING BY STRING DIST: {pl} -> {best_cand}")
        valid_player_names[(pl, team, season)] = best_cand
        return True

    # by previous season
    try:
        if valid_player_names[(pl, team, season - 1)] in lineup:
            print(f"FOUND PAIRING BY PREVIOUS SEASON: {pl} -> {valid_player_names[(pl, team, season - 1)]}")
            valid_player_names[(pl, team, season)] = valid_player_names[(pl, team, season - 1)]
            return True
    except KeyError:
        pass

    # by database mistake
    if valid_player_names.get((pl, other_team, season), pl) in other_lineup:
        name = valid_player_names.get((pl, other_team, season), pl)
        # print(pl, name, team, other_team)
        ev = events.loc[(events['id_odsp'] == match_id) & (events["player"] == name) & (events["event_team"] == team)]
        if len(ev) == 1:
            vals = ev.values.reshape(-1)
            events.loc[ev.index[0]] = vals[
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
            print(f"FOUND MISTAKE WHERE {name} SHOULD BELONG TO TEAM {other_team}")
            return True

        ev = events.loc[
            (events['id_odsp'] == match_id) & (events["player"] == name) & (events["event_team"] == other_team)]

        if len(ev) == 1:
            vals = ev.values.reshape(-1)
            events.loc[ev.index[0]] = vals[
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
            print(f"FOUND MISTAKE WHERE {name} SHOULD BELONG TO TEAM {other_team}")
            return True

        ev = events.loc[(events['id_odsp'] == match_id) & (events["player2"] == name) & (events["event_team"] == team)]
        if len(ev) == 1:
            vals = ev.values.reshape(-1)
            events.loc[ev.index[0]] = vals[
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
            print(f"FOUND MISTAKE WHERE {name} SHOULD BELONG TO TEAM {other_team}")
            if nantest(ev["player"].values.reshape(-1)[0]):
                events.loc[ev.index[0], "player"] = name
            return True

        ev = events.loc[
            (events['id_odsp'] == match_id) & (events["player2"] == name) & (events["event_team"] == other_team)]
        if len(ev) == 1:
            vals = ev.values.reshape(-1)
            events.loc[ev.index[0]] = vals[
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
            print(f"FOUND MISTAKE WHERE {name} SHOULD BELONG TO TEAM {other_team}")
            if nantest(ev["player"].values.reshape(-1)[0]):
                events.loc[ev.index[0], "player"] = name
            return True

        # while True:
        #     print(f"PLAYER {name} POSSIBLY BELONGING TO TEAM {other_team}. IS THAT TRUE?")
        #     ans = input("Y for yes, N for No")
        #     if ans == "Y":
        #         return True
        #     if ans == "N":
        #         return False
        print(f"PLAYER {name} BELONGS TO TEAM {other_team}")
        return True

    # by manual input
    return False


def nantest(x):
    return x != x


def fit_rest_names(players_from_events, lineup, other_lineup, team, other_team, season, match_id,
                   dist_f=distance.levenshtein):
    set_to_lineup, lineup_to_set = {}, {}

    for pl_full in players_from_events:
        for potential_name_full in lineup:
            pl, potential_name = pl_full.split(), potential_name_full.split()

            if len(pl) > len(potential_name) and check_fit(pl, potential_name):
                append_to_dictionary(set_to_lineup, pl_full, potential_name_full)
                append_to_dictionary(lineup_to_set, potential_name_full, pl_full)
            if len(pl) < len(potential_name) and check_fit(potential_name, pl):
                append_to_dictionary(set_to_lineup, pl_full, potential_name_full)
                append_to_dictionary(lineup_to_set, potential_name_full, pl_full)

    for pl in players_from_events:
        dists = sorted([(dist_f(pl, candidate, normalized=True), candidate) for candidate in lineup])
        if not resolve_match(pl, lineup, other_lineup, team, other_team, season, match_id, set_to_lineup, lineup_to_set,
                             dists):
            print(f"Conflict detected!")
            print(team, "|", other_team, "|", match_id)
            print(f"{pl}: ")
            for score, candidate in dists:
                print(f"{candidate:30} - {score:.5f}")
            print()

            cand = input("Player name or E to finish.").strip()
            if cand == "" or cand == "E":
                return True
            print(f"FOUND PAIRING BY INPUT: {pl} -> {cand}")
            valid_player_names[(pl, team, season)] = cand

    return False


def build_player_name_translator(LEFT_AT=0):
    # temp_db = ginf.loc[ginf["league"] == "I1"].reset_index(drop=True)

    # for ind, match in temp_db.loc[LEFT_AT:].iterrows(): #tymczasowo dopóki nie ma całego lineups
    for ind, match in ginf.loc[LEFT_AT:].iterrows():
        match_id = match['id_odsp']
        home_team = match['ht']
        away_team = match['at']
        date = match['date']
        season = match['season']

        print(f"{home_team} vs {away_team} on {date} ({season})")

        if not match["adv_stats"]:  # no data in events
            print("No data available")
            continue  # pass, continue, cokolwiek

        lineup = lineups.loc[
            (lineups['date'] == date) & (lineups['home_team'] == home_team) & (lineups['away_team'] == away_team)]
        # no duplicates
        try:
            assert len(lineup) == 1
        except AssertionError as e:
            print(ind)
            raise e

        home_lineup_db = lineup[[f"hp{i}" for i in range(1, 12)] + [f"hs{i}" for i in range(1, 13)]].values[0]
        away_lineup_db = lineup[[f"ap{i}" for i in range(1, 12)] + [f"as{i}" for i in range(1, 13)]].values[0]

        # filter out nans
        home_lineup = set(filter(lambda v: v == v, home_lineup_db))
        away_lineup = set(filter(lambda v: v == v, away_lineup_db))

        # Assumption: there are no two players of the same name
        assert len(home_lineup) == len(list(filter(lambda v: v == v, home_lineup_db)))
        assert len(away_lineup) == len(list(filter(lambda v: v == v, away_lineup_db)))

        evs_in_match = events.loc[events['id_odsp'] == match_id]
        ht_player_set, at_player_set = set(), set()
        for _, ev in evs_in_match.iterrows():
            ev_team, op_team = ev['event_team'], ev['opponent']

            if ev_team == home_team:
                add_players_to_set(ev, ht_player_set, at_player_set)
            else:
                add_players_to_set(ev, at_player_set, ht_player_set)

        ht_unmatched = filter_out_correct(ht_player_set, home_lineup, home_team, season)
        at_unmatched = filter_out_correct(at_player_set, away_lineup, away_team, season)

        # print(ht_unmatched)
        # print(at_unmatched)
        try:
            if (fit_rest_names(ht_unmatched, home_lineup, away_lineup, home_team, away_team, season, match_id) or
                    fit_rest_names(at_unmatched, away_lineup, home_lineup, away_team, home_team, season, match_id)):
                with open("valid_player_names.pkl", "wb") as f:
                    pickle.dump(valid_player_names, f)
                return ind
        except BaseException as e:
            print(ind)
            raise e

        if ind % 10 == 0:
            with open("valid_player_names.pkl", "wb") as f:
                pickle.dump(valid_player_names, f)

    with open("valid_player_names.pkl", "wb") as f:
        pickle.dump(valid_player_names, f)


build_player_name_translator()
