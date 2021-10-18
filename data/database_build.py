import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import itertools

from sklearn.base import BaseEstimator, TransformerMixin

lineups = pd.read_csv("lineups.csv")
events = pd.read_csv("events.csv")
ginf = pd.read_csv('ginf.csv')
name_to_url = pd.read_pickle('name_to_url.pkl')
url_to_name = pd.read_pickle('url_to_name.pkl')
valid_player_names = pd.read_pickle('valid_player_names.pkl')

labels = ["date", "home_team", "away_team"]
lineups = lineups.drop_duplicates(subset=labels, keep='first')
player_pos = pd.read_pickle('player_pos.pkl')
table_pos = pd.read_pickle('table_pos.pkl')

EVENTS_DICT = {1: "attempts(shots)",
               2: "corners",
               3: "fouls",
               4: "yellow_cards",
               5: "second_yellow_card",
               6: "straight_red_card",
               7: "substitutions",
               8: "free_kicks",
               9: "offsides",
               10: "hand_ball",
               11: "penalties",
               12: "key_passes",
               13: "offside_pass",
               15: "own_goals"}

LOCATIONS_DICT = {1: "middle",
                  2: "difficult",
                  3: "easy",
                  4: "wing",
                  5: "wing",
                  6: "difficult",
                  7: "difficult",
                  8: "difficult",
                  9: "box_side",
                  10: "smaller_box_side",
                  11: "box_side",
                  12: "smaller_box_side",
                  13: "easy",
                  14: "easy",
                  15: "middle",
                  16: "difficult",
                  17: "difficult",
                  18: "difficult"}

SITUATIONS_DICT = {1: "Open_play",
                   2: "Set_piece",
                   3: "Corner",
                   4: "Free_kick"}


def isnull(x):
    return x != x


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_save(path, ob):
    with open(path, "wb") as f:
        pickle.dump(ob, f)


t_cols = ["name", "season", "round", "date",
          "games_played", "games_played_home", "games_played_away",
          "goal_scored", "goal_scored_home", "goal_scored_away",
          "goal_conceded", "goal_conceded_home", "goal_conceded_away",
          "own_goals", "attempts(shots)", "shots_on_target",
          "fouls", "free_kicks_won", "free_kicks_conceded", "corners_won",
          "corners_conceded", "penalties_won", "penalties_conceded",
          "offsides", "yellow_cards", "red_cards",
          "substitutions", "Open_play_made", "Open_play_conceded", "Set_piece_made", "Set_piece_conceded",
          "Corner_made", "Corner_conceded", "Free_kick_made", "Free_kick_conceded", "shot_made_middle",
          "shot_conceded_middle",
          "shot_made_difficult", "shot_conceded_difficult", "shot_made_easy", "shot_conceded_easy", "shot_made_wing",
          "shot_conceded_wing",
          "shot_made_smaller_box_side", "shot_conceded_smaller_box_side", "shot_made_box_side",
          "shot_conceded_box_side"]
p_cols = ["id", "name", "season", "round", "date", "team",
          "goal_scored", "goal_scored_home", "goal_scored_away",
          "attempts(shots)", "corners_conceded", "fouls",
          "yellow_cards", "second_yellow_card", "straight_red_card",
          "free_kicks_won", "offsides", "key_passes",
          "time_played", "subs_in", "subs_out",
          "hand_ball", "offside_pass", "has_injured",
          "own_goals", "shots_on_target", "days_without_injury", "has_played"]
g_cols = ["id", "name", "season", "round", "date", "team",
          "fouls", "yellow_cards",
          "second_yellow_card", "straight_red_card",
          "has_injured", "shots_saved_percent",
          "shots_saved", "goal_conceded", "goal_conceded_home",
          "goal_conceded_away", "subs_in", "subs_out",
          "time_played", "shots_on_target", "own_goals", "days_without_injury", "has_played"]

teams = pd.DataFrame(columns=t_cols)

TEAM_STAT_NUMBER = len(teams.columns) - 4
players = pd.DataFrame(columns=p_cols)
PLAYER_STAT_NUMBER = len(players.columns) - 6

goalkeepers = pd.DataFrame(columns=g_cols)
GOALKEEPER_STAT_NUMBER = len(goalkeepers.columns) - 6

player_round_cache = {}
team_round_cache = {}


def get_round_number(name, season):
    global team_round_cache
    return team_round_cache.get((name, season), 0) + 1


def get_player_round(id, season):
    global player_round_cache
    return player_round_cache.get((id, season), 0) + 1


def init_row(df, name, season, round, stat_number, id=None, date=None, team=None):
    prev_record = [0] * stat_number
    df.loc[len(df)] = [name, season, round, date] + prev_record if id is None else [id, name, season, round, date,
                                                                                    team] + prev_record


def append_value(df, name, season, round, column, home_or_away=None, value=1, id=None, time=False):
    if id is None:
        df.loc[(df["name"] == name) & (df["season"] == season) & (df["round"] == round), column] += value
    elif not time:
        df.loc[(df["id"] == id) & (df["season"] == season) & (df["round"] == round), column] += value
    else:
        row = df.loc[(df["id"] == id) & (df["season"] == season) & (df["round"] == round)]
        if row[column].values[0] == 0:
            df.loc[(df["id"] == id) & (df["season"] == season) & (df["round"] == round), column] = value
        else:
            df.loc[(df["id"] == id) & (df["season"] == season) & (df["round"] == round), column] = value - row[column]

    if home_or_away is not None:
        if id is None:
            df.loc[((df["name"] == name) & (df["season"] == season) &
                    (df["round"] == round)), f"{column}_{home_or_away}"] += value
        else:
            df.loc[((df["id"] == id) & (df["season"] == season) &
                    (df["round"] == round)), f"{column}_{home_or_away}"] += value


def init_players_in_match(lineup, season, date, _goalkeepers, _players):
    rounds = {}
    for nr, team in list(itertools.product([i for i in range(1, 13)], ["ap", "hp", "hs", "as"])):
        pl = team + str(nr)
        if nr == 12 and (team == "ap" or team == "hp"):
            pass
        elif not isnull(lineup[pl]):
            team_name = "home_team" if team[0] == "h" else "away_team"
            pl_id = name_to_url[(lineup[pl].values[0], season, lineup[team_name].values[0])]
            pl_role = player_pos[pl_id]
            (db, stats) = (_goalkeepers, GOALKEEPER_STAT_NUMBER) if pl_role == "Goalkeeper" else (
                _players, PLAYER_STAT_NUMBER)
            if pl_role == "Goalkeeper":
                rnd = get_player_round(pl_id, season)
            else:
                rnd = get_player_round(pl_id, season)
            rounds[(lineup[pl].values[0], season, lineup[team_name].values[0])] = rnd
            init_row(db, lineup[pl].values[0], season, rnd, stats, pl_id, date, lineup[team_name].values[0])
    return rounds


def update_time_played(lineup, season, rounds, _goalkeepers, _players):
    for nr, team in list(itertools.product([i for i in range(1, 13)], ["ap", "hp", "hs", "as"])):
        pl = team + str(nr)
        if nr == 12 and (team == "ap" or team == "hp"):
            pass
        elif not isnull(lineup[pl]):
            team_name = "home_team" if team[0] == "h" else "away_team"

            pl_id = name_to_url[(lineup[pl].values[0], season, lineup[team_name].values[0])]
            pl_role = player_pos[pl_id]
            pl_round = rounds[(lineup[pl].values[0], season, lineup[team_name].values[0])]
            db = _goalkeepers if pl_role == "Goalkeeper" else _players
            row = db.loc[(db["id"] == pl_id) & (db["season"] == season) & (db["round"] == pl_round)]
            if len(row) > 0:
                # zabezpieczenie co do tych którzy zeszli z boiska za kartki
                if row["subs_out"].values[0] != 0 and (
                        row["second_yellow_card"].values[0] == 1 or row["straight_red_card"].values[0] == 1):
                    db.loc[(db["id"] == pl_id) & (db["season"] == season) & (db["round"] == pl_round), "subs_out"] = 0

                if row["subs_in"].values[0] == 0 and row["subs_out"].values[0] == 0 and team[1] == "p":
                    db.loc[
                        (db["id"] == pl_id) & (db["season"] == season) & (db["round"] == pl_round), "time_played"] = 90
                elif row["subs_in"].values[0] != 0 and row["subs_out"].values[0] == 0:
                    db.loc[(db["id"] == pl_id) & (db["season"] == season) & (
                            db["round"] == pl_round), "time_played"] = 90 - row["time_played"]

                if db.loc[(db["id"] == pl_id) & (db["season"] == season) &
                          (db["round"] == pl_round), "time_played"].values[0] != 0:
                    db.loc[(db["id"] == pl_id) & (db["season"] == season) & (db["round"] == pl_round), "has_played"] = 1


def count_days(date1, date2):
    year1, month1, day1 = date1[0:4], date1[5:7], date1[8:]
    year2, month2, day2 = date2[0:4], date2[5:7], date2[8:]
    d1 = datetime(int(year1), int(month1), int(day1))
    d2 = datetime(int(year2), int(month2), int(day2))
    return (d2 - d1).days


def update_injuries(lineup, season, rounds, _goalkeepers, _players):
    for nr, team in list(itertools.product([i for i in range(1, 13)], ["ap", "hp", "hs", "as"])):
        pl = team + str(nr)
        if nr == 12 and (team == "ap" or team == "hp"):
            pass
        elif not isnull(lineup[pl]):
            team_name = "home_team" if team[0] == "h" else "away_team"

            pl_id = name_to_url[(lineup[pl].values[0], season, lineup[team_name].values[0])]
            pl_role = player_pos[pl_id]
            pl_round = rounds[(lineup[pl].values[0], season, lineup[team_name].values[0])]
            db = _goalkeepers if pl_role == "Goalkeeper" else _players
            row = db.loc[(db["id"] == pl_id) & (db["season"] == season) & (db["round"] == pl_round)]
            if len(row) > 0:
                if row["has_injured"].values[0] == 1:
                    # row["days_without_injury"] = 0
                    db.loc[(db["id"] == pl_id) & (db["season"] == season) & (
                            db["round"] == pl_round), "days_without_injury"] = 0
                else:
                    if pl_round == 1:
                        # row["days_without_inury"] = 10000
                        db.loc[(db["id"] == pl_id) & (db["season"] == season) & (
                                db["round"] == pl_round), "days_without_injury"] = 10000
                        # print("ustawione")
                    else:
                        if pl_role == "Goalkeeper":
                            prev = goalkeepers.loc[(goalkeepers["id"] == pl_id) & (goalkeepers["season"] == season) & (
                                    goalkeepers["round"] == pl_round - 1)]
                        else:
                            prev = players.loc[(players["id"] == pl_id) & (players["season"] == season) & (
                                    players["round"] == pl_round - 1)]
                        if prev["days_without_injury"].values[0] == 10000:
                            db.loc[(db["id"] == pl_id) & (db["season"] == season) & (
                                    db["round"] == pl_round), "days_without_injury"] = 10000
                        else:
                            days = count_days(prev["date"].values[0], row["date"].values[0])
                            db.loc[(db["id"] == pl_id) & (db["season"] == season) & (
                                    db["round"] == pl_round), "days_without_injury"] = \
                                prev["days_without_injury"].values[0] + days


def keeper_update(season, rounds, team, lineup, _goalkeepers, saved=False, goal_con=False):
    l_team = "h" if team == lineup["home_team"].values[0] else "a"
    st_keeper_id = name_to_url[(lineup[l_team + "p1"].values[0], season, team)]
    st_keeper_round = rounds[(lineup[l_team + "p1"].values[0], season, team)]
    starting_keeper = _goalkeepers.loc[
        (_goalkeepers["id"] == st_keeper_id) & (_goalkeepers["round"] == st_keeper_round) & (
                _goalkeepers["season"] == season)]
    if starting_keeper["subs_out"].values[0] == 0:
        if goal_con:
            hoa = "home" if l_team == "h" else "away"
            append_value(_goalkeepers, lineup[l_team + "p1"].values[0], season, st_keeper_round, "goal_conceded",
                         home_or_away=hoa, id=st_keeper_id)
            append_value(_goalkeepers, lineup[l_team + "p1"].values[0], season, st_keeper_round, "shots_on_target",
                         id=st_keeper_id)
        if saved:
            append_value(_goalkeepers, lineup[l_team + "p1"].values[0], season, st_keeper_round, "shots_saved",
                         id=st_keeper_id)
            append_value(_goalkeepers, lineup[l_team + "p1"].values[0], season, st_keeper_round, "shots_on_target",
                         id=st_keeper_id)
    else:
        for i in range(1, 13):
            pl = l_team + "s" + str(i)
            if isnull(lineup[pl].values[0]):
                return

            if (lineup[pl].values[0], season, team) in name_to_url:
                # print(lineup[pl].values[0], season, team, pl)
                pl_id = name_to_url[(lineup[pl].values[0], season, team)]
            if player_pos[pl_id] == "Goalkeeper":
                pl_round = rounds[(lineup[pl].values[0], season, team)]
                gk = _goalkeepers.loc[(_goalkeepers["id"] == pl_id) & (_goalkeepers["season"] == season) & (
                        _goalkeepers["round"] == pl_round)]
                if gk["subs_in"].values[0] != 0 and gk["subs_out"].values[0] == 0:
                    if goal_con:
                        hoa = "home" if l_team == "h" else "away"
                        append_value(_goalkeepers, lineup[pl].values[0], season, pl_round, "goal_conceded",
                                     home_or_away=hoa, id=pl_id)
                        append_value(_goalkeepers, lineup[pl].values[0], season, pl_round, "shots_on_target", id=pl_id)
                    if saved:
                        append_value(_goalkeepers, lineup[pl].values[0], season, pl_round, "shots_saved", id=pl_id)
                        append_value(_goalkeepers, lineup[pl].values[0], season, pl_round, "shots_on_target", id=pl_id)


def postprocess(lineup, season, rounds, _goalkeepers, _players, _teams):
    global players
    global teams
    global goalkeepers
    global player_round_cache
    update_time_played(lineup, season, rounds, _goalkeepers, _players)
    update_injuries(lineup, season, rounds, _goalkeepers, _players)
    players = pd.concat([players, _players])
    teams = pd.concat([teams, _teams])
    goalkeepers = pd.concat([goalkeepers, _goalkeepers])
    pls = [f"hp{i}" for i in range(1, 12)] + [f"hs{i}" for i in range(1, 13)] + [f"ap{i}" for i in range(1, 12)] + [
        f"as{i}" for i in range(1, 13)]
    for p in pls:
        if not isnull(lineup[p].values[0]):
            team = lineup["home_team"].values[0] if p[0] == "h" else lineup["away_team"].values[0]
            pl_id = name_to_url[(lineup[p].values[0], season, team)]
            player_round_cache[(pl_id, season)] = rounds[(lineup[p].values[0], season, team)]


def find_name(pl, team, other_team, lineup, other_lineup, season):
    name = valid_player_names.get((pl, team, season), pl)
    if name in lineup:
        # print(f"found {name} in first team: {team}")
        return name, team, other_team, lineup, other_lineup
    name = valid_player_names.get((pl, other_team, season), pl)
    if name in other_lineup:
        # print(f"found {name} in second team: {other_team}")
        return name, other_team, team, other_lineup, lineup
    print(f"{team}: {lineup}")
    print(other_team, other_lineup)
    raise TypeError(f"={valid_player_names.get((pl, team, season), pl)}= ={name}= ={pl}= {season}")


def team_stats(match_id, season, home_team, away_team, date):
    evs = events.loc[events["id_odsp"] == match_id, :]

    _teams = pd.DataFrame(columns=t_cols)
    _players = pd.DataFrame(columns=p_cols)
    _goalkeepers = pd.DataFrame(columns=g_cols)

    hm_round = get_round_number(home_team, season)
    init_row(_teams, home_team, season, hm_round, TEAM_STAT_NUMBER, date=date)
    aw_round = get_round_number(away_team, season)
    init_row(_teams, away_team, season, aw_round, TEAM_STAT_NUMBER, date=date)

    lineup = lineups.loc[
        (lineups["date"] == date) & (lineups["home_team"] == home_team) & (lineups["away_team"] == away_team)]
    player_rounds = init_players_in_match(lineup, season, date, _goalkeepers, _players)

    home_lineup = lineup[[f"hp{i}" for i in range(1, 12)] + [f"hs{i}" for i in range(1, 13)]].values[0]
    away_lineup = lineup[[f"ap{i}" for i in range(1, 12)] + [f"as{i}" for i in range(1, 13)]].values[0]

    append_value(_teams, home_team, season, hm_round, "games_played", "home")
    append_value(_teams, away_team, season, aw_round, "games_played", "away")

    for _, ev in evs.iterrows():
        # print("-----------------new ev---------------------")
        ev_team, op_team = ev["event_team"], ev["opponent"]
        ev_home_or_away = "home" if home_team == ev_team else "away"
        ev_round = hm_round if home_team == ev_team else aw_round
        op_home_or_away = "away" if home_team == ev_team else "home"
        op_round = aw_round if home_team == ev_team else hm_round

        if ev_home_or_away == "home":
            ev_lineup, op_lineup = home_lineup, away_lineup
        else:
            ev_lineup, op_lineup = away_lineup, home_lineup

        ev_player = ev["player"]
        ev_player2 = ev["player2"]
        ev_plin = ev["player_in"]
        ev_plout = ev["player_out"]
        ev_pl_name, ev_pl2_name, ev_plin_name, ev_plout_name = "", "", "", ""

        if not isnull(ev_player):
            if ev["event_type"] != 2 and ev["event_type2"] != 15:
                ev_pl_name, ev_team, op_team, ev_lineup, op_lineup = find_name(ev_player, ev_team, op_team, ev_lineup,
                                                                               op_lineup, season)
            else:
                # print(ev_team, op_team)
                ev_pl_name, op_team, ev_team, op_lineup, ev_lineup = find_name(ev_player, op_team, ev_team, op_lineup,
                                                                               ev_lineup, season)
                # print(f"{ev_team} {op_team}")
                # print()
        if not isnull(ev_player2):
            if ev["event_type"] != 2:
                ev_pl2_name, ev_team, op_team, ev_lineup, op_lineup = find_name(ev_player2, ev_team, op_team, ev_lineup,
                                                                                op_lineup, season)
        if not isnull(ev_plin):
            ev_plin_name, ev_team, op_team, ev_lineup, op_lineup = find_name(ev_plin, ev_team, op_team, ev_lineup,
                                                                             op_lineup, season)
        if not isnull(ev_plout):
            ev_plout_name, ev_team, op_team, ev_lineup, op_lineup = find_name(ev_plout, ev_team, op_team, ev_lineup,
                                                                              op_lineup, season)

        if ev["event_type"] == 7:
            if ev_plin_name != "":
                try:
                    ev_plin_id = name_to_url[(ev_plin_name, season, ev_team)]
                    ev_plin_round = player_rounds[(ev_plin_name, season, ev_team)]
                    plin_db = _goalkeepers if player_pos[ev_plin_id] == "Goalkeeper" else _players
                except KeyError as e:
                    print("---subs in---")
                    print((ev_plin_name, season, ev_team, ev_plin_id, ev["event_type"], ev["event_type2"],
                           ev["id_event"], ev_plin))
                    raise e
            if ev_plout_name != "":
                try:
                    ev_plout_id = name_to_url[(ev_plout_name, season, ev_team)]
                    ev_plout_round = player_rounds[(ev_plout_name, season, ev_team)]
                    plout_db = _goalkeepers if player_pos[ev_plout_id] == "Goalkeeper" else _players
                except KeyError as e:
                    print("---subs out---")
                    print((ev_plout_name, season, ev_team, ev_plin_id, ev["event_type"], ev["event_type2"],
                           ev["id_event"], ev_plout))
                    raise e
        elif ev['event_type2'] == 15 or ev['event_type'] == 2:
            if ev_pl_name != "":
                try:
                    ev_pl_id = name_to_url[(ev_pl_name, season, op_team)]
                    ev_pl_round = player_rounds[(ev_pl_name, season, op_team)]
                    db = _goalkeepers if player_pos[ev_pl_id] == "Goalkeeper" else _players
                except KeyError as e:
                    print("---own goal / corner---")
                    print((ev_pl_name, season, op_team, ev_pl_id, ev["event_type"], ev["event_type2"], ev["id_event"],
                           ev_player))
                    raise e
        else:
            if ev_pl_name != "":
                try:
                    ev_pl_id = name_to_url[(ev_pl_name, season, ev_team)]
                    ev_pl_round = player_rounds[(ev_pl_name, season, ev_team)]
                    db = _goalkeepers if player_pos[ev_pl_id] == "Goalkeeper" else _players
                except KeyError as e:
                    print("---other events---")
                    print((ev_pl_name, season, ev_team, ev_pl_id, ev["event_type"], ev["event_type2"], ev["id_event"],
                           ev_player))
                    raise e

        evapp2 = False
        if (ev_pl2_name, season, ev_team) in name_to_url and (ev_pl2_name, season, ev_team) in player_rounds:
            ev_pl2_id = name_to_url[(ev_pl2_name, season, ev_team)]
            ev_pl2_round = player_rounds[(ev_pl2_name, season, ev_team)]
            db2 = _goalkeepers if player_pos[ev_pl2_id] == "Goalkeeper" else _players
            evapp2 = True
        else:
            evapp2 = False

        # goal scored
        if ev['is_goal'] == 1:
            append_value(_teams, ev_team, season, ev_round, "goal_scored", ev_home_or_away)
            append_value(_teams, op_team, season, op_round, "goal_conceded", op_home_or_away)
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[1])
            if not isnull(ev['location']) and ev['location'] != 19:
                append_value(_teams, ev_team, season, ev_round, "shot_made_" + LOCATIONS_DICT[ev['location']])
                append_value(_teams, op_team, season, op_round, "shot_conceded_" + LOCATIONS_DICT[ev['location']])

            if not isnull(ev['situation']):
                append_value(_teams, ev_team, season, ev_round, SITUATIONS_DICT[ev['situation']] + "_made")
                append_value(_teams, op_team, season, op_round, SITUATIONS_DICT[ev['situation']] + "_conceded")

            keeper_update(season, player_rounds, op_team, lineup, _goalkeepers, goal_con=True)

            # own goal
            if ev['event_type2'] == 15:
                # adding to team
                append_value(_teams, op_team, season, op_round, EVENTS_DICT[15])
                # adding to player
                if player_pos[ev_pl_id] != "Goalkeeper":
                    append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[15], id=ev_pl_id)
            else:
                if player_pos[ev_pl_id] != "Goalkeeper":
                    append_value(db, ev_pl_name, season, ev_pl_round, "goal_scored", ev_home_or_away, id=ev_pl_id)

                    # key pass
        if ev['event_type2'] == 12:
            # player key pass++
            if evapp2 and player_pos[ev_pl2_id] != "Goalkeeper":
                append_value(db2, ev_pl2_name, season, ev_pl2_round, EVENTS_DICT[12], id=ev_pl2_id)
        # offside pass
        elif ev['event_type2'] == 13:
            # player offside pass++
            if evapp2 and player_pos[ev_pl2_id] != "Goalkeeper":
                append_value(db2, ev_pl2_name, season, ev_pl2_round, EVENTS_DICT[13], id=ev_pl2_id)

        # attempt
        if ev['event_type'] == 1:
            # player attempt++
            if player_pos[ev_pl_id] != "Goalkeeper":
                append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[1], id=ev_pl_id)

            # attempts(shots)
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[1])

            # shot at given location
            if not isnull(ev['location']) and ev['location'] != 19:
                append_value(_teams, ev_team, season, ev_round, "shot_made_" + LOCATIONS_DICT[ev['location']])
                append_value(_teams, op_team, season, op_round, "shot_conceded_" + LOCATIONS_DICT[ev['location']])

            if not isnull(ev['situation']):
                append_value(_teams, ev_team, season, ev_round, SITUATIONS_DICT[ev['situation']] + "_made")
                append_value(_teams, op_team, season, op_round, SITUATIONS_DICT[ev['situation']] + "_conceded")

                # shot on target
            if ev['shot_outcome'] == 1:
                # team
                append_value(_teams, ev_team, season, ev_round, "shots_on_target")
                # player
                if player_pos[ev_pl_id] != "Goalkeeper":
                    append_value(db, ev_pl_name, season, ev_pl_round, "shots_on_target", id=ev_pl_id)
                    # shot saved
                    if ev['is_goal'] == 0:
                        saved = True
                    else:
                        saved = False
                    keeper_update(season, player_rounds, op_team, lineup, _goalkeepers, saved)

        # corner
        elif ev['event_type'] == 2:
            # teams
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[2] + "_won")
            append_value(_teams, op_team, season, op_round, EVENTS_DICT[2] + "_conceded")
            # player corners conceded ++
            if player_pos[ev_pl_id] != "Goalkeeper":
                append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[2] + "_conceded", id=ev_pl_id)

        # fouls
        elif ev['event_type'] == 3:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[3])
            # player fouls++
            append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[3], id=ev_pl_id)
        # yellow card
        elif ev['event_type'] == 4:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[4])
            # player yellow card++
            append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[4], id=ev_pl_id)
        # second yellow card
        elif ev['event_type'] == 5:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[4])
            append_value(_teams, ev_team, season, ev_round, "red_cards")
            # player second yellow++
            append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[5], id=ev_pl_id)
            append_value(db, ev_pl_name, season, ev_pl_round, "subs_out", id=ev_pl_id)
            append_value(db, ev_pl_name, season, ev_pl_round, "time_played", value=ev['time'], id=ev_pl_id, time=True)

        # red card
        elif ev['event_type'] == 6:
            append_value(_teams, ev_team, season, ev_round, "red_cards")
            # player red++
            append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[6], id=ev_pl_id)
            append_value(db, ev_pl_name, season, ev_pl_round, "subs_out", id=ev_pl_id)
            append_value(db, ev_pl_name, season, ev_pl_round, "time_played", value=ev['time'], id=ev_pl_id, time=True)

        # substitution
        elif ev['event_type'] == 7:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[7])
            # player substitution += minutes
            # player in
            # print(ev["id_event"])
            append_value(plin_db, ev_plin_name, season, ev_plin_round, "subs_in", id=ev_plin_id)
            append_value(plin_db, ev_plin_name, season, ev_plin_round, "time_played", value=ev['time'], id=ev_plin_id,
                         time=True)
            # player out
            append_value(plout_db, ev_plout_name, season, ev_plout_round, "subs_out", id=ev_plout_id)
            append_value(plout_db, ev_plout_name, season, ev_plout_round, "time_played", value=ev['time'],
                         id=ev_plout_id, time=True)
            injury = ev['text'].find("injur")
            if injury != -1:
                append_value(plout_db, ev_plout_name, season, ev_plout_round, "has_injured", id=ev_plout_id)

        # free kick won
        elif ev['event_type'] == 8:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[8] + "_won")
            append_value(_teams, op_team, season, op_round, EVENTS_DICT[8] + "_conceded")
            # player free kicks won++
            if player_pos[ev_pl_id] != "Goalkeeper":
                append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[8] + "_won", id=ev_pl_id)

        # offside
        elif ev['event_type'] == 9:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[9])
            # player offsides ++
            if player_pos[ev_pl_id] != "Goalkeeper":
                # print("\n"+player_pos[ev_pl_id] + " | " + ev_pl_id)
                append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[9], id=ev_pl_id)

        # handball
        elif ev['event_type'] == 10:
            # player hand ball ++
            if player_pos[ev_pl_id] != "Goalkeeper":
                append_value(db, ev_pl_name, season, ev_pl_round, EVENTS_DICT[10], id=ev_pl_id)
        # penalty
        elif ev['event_type'] == 11:
            append_value(_teams, ev_team, season, ev_round, EVENTS_DICT[11] + "_conceded")
            append_value(_teams, op_team, season, op_round, EVENTS_DICT[11] + "_won")

    postprocess(lineup, season, player_rounds, _goalkeepers, _players, _teams)
    team_round_cache[(home_team, season)] = hm_round
    team_round_cache[(away_team, season)] = aw_round


def build(start_at=0):
    for ind, row in tqdm(ginf.loc[start_at:].iterrows(), total=len(ginf) - start_at):
        if row["adv_stats"]:
            try:
                id_if_error = row["id_odsp"]
                team_stats(row["id_odsp"], row["season"], row["ht"], row["at"], row["date"])
            except BaseException as e:
                print(f"iters: {start_at}, id_if_error: {id_if_error}")
                raise e


def fix_mistakes():
    global ginf
    ginf = ginf.loc[(ginf["season"] != 2012) | (ginf["league"] != "E0")]
    ginf = ginf.loc[(ginf["season"] != 2013) | (ginf["league"] != "E0")]
    ginf = ginf.loc[((ginf["ht"] != "Kaiserslautern") & (ginf["at"] != "Kaiserslautern")) | (ginf["season"] != 2013)]

    lineups.loc[
        (lineups["home_team"] == "Valencia") & (lineups["away_team"] == "Racing Santander"), "hs5"] = "alberto costa "

    lineups.loc[(lineups["home_team"] == "Granada") & (lineups["away_team"] == "Mallorca") & (
            lineups["season"] == 2012), "hp1"] = "roberto"
    goalkeepers.loc[goalkeepers["name"] == "roberto fernandez", "name"] = "roberto"

    lineups.loc[(lineups["home_team"] == "Granada") & (lineups["away_team"] == "Mallorca") & (
            lineups["season"] == 2012), "hp11"] = "alexandre geijo"
    players.loc[players["name"] == "alex geijo", "name"] = "alexandre geijo"

    lineups.loc[(lineups["home_team"] == "Granada") & (lineups["away_team"] == "Mallorca") & (
            lineups["season"] == 2012), "ap4"] = "chico flores"
    players.loc[players["name"] == "chico", "name"] = "chico flores"

    lineups.loc[(lineups["home_team"] == "Granada") & (lineups["away_team"] == "Mallorca") & (
            lineups["season"] == 2012), "ap6"] = "jose marti"
    players.loc[players["name"] == "josep lluis marti", "name"] = "jose marti"

    lineups.loc[(lineups["home_team"] == "Granada") & (lineups["away_team"] == "Mallorca") & (
            lineups["season"] == 2012), "ap9"] = "gonzalo castro irizabal"
    players.loc[players["name"] == "chory castro", "name"] = "gonzalo castro irizabal"

    name_to_url[("xisco hernandez", 2012, "Mallorca")] = "/players/francisco-hernandez-marcos/153360/"

    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "hp5"] = "alvaro pereira barragan"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "hp6"] = "esteban cambiasso delau"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "hp7"] = "fredy guarin vasquez"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "hp8"] = "walter gargano guevara"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "hs5"] = "walter samuel lujan "
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "ap4"] = "luis carlos"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "as6"] = "ribair rodriguez perez"
    lineups.loc[(lineups["home_team"] == "Internazionale") & (lineups["away_team"] == "Siena") & (
            lineups["season"] == 2013), "as11"] = "michele paolucci"

    lineups.loc[(lineups["home_team"] == "Catania") & (lineups["away_team"] == "Napoli") & (
            lineups["season"] == 2013), "hp2"] = "pablo valeira alvarez"
    lineups.loc[(lineups["home_team"] == "AS Roma") & (lineups["away_team"] == "Parma") & (
            lineups["season"] == 2014), "hp5"] = "vassilis torosidis"


build()
fix_mistakes()


# group_by_position - bool
# n_games - None or int
# exact_results - bool
# n_splits - [2, ...]
class DatabaseBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, n_games=None, exact_results=False):
        self.n_games = n_games
        self.exact_results = exact_results

        self._team_history = {}
        self._player_history = {}

    def fit(self, X, y=None):
        teams, players, keepers = X

        self.team_stats_ = teams.columns[4:]
        self.player_stat_ = players.columns[6:]
        self.keeper_stat_ = keepers.columns[6:]

        self.columns_ = []

        if self.n_games is None:
            form_tab = [None]
        else:
            form_tab = ["total", f"last_{self.n_games}"]

        for where, first_XI, pos in itertools.product(["home", "away"], [True, False],
                                                      ["Team", 'Goalkeeper', 'Defender', 'Midfielder', 'Attacker']):
            for form in form_tab:

                if pos == 'Team':
                    stats = self.team_stats_
                elif pos == 'Goalkeeper':
                    stats = self.keeper_stat_
                else:
                    stats = self.player_stat_

                if form is None:
                    prefix = f"{where}_{pos}"
                else:
                    prefix = f"{where}_{pos}_{form}"

                for stat in stats:
                    if first_XI:
                        self.columns_.append(f"{prefix}_{stat}")
                    else:
                        if pos != "Team":
                            self.columns_.append(f"{prefix}_sub_{stat}")

                for res in ["wins", "draws", "lost"]:
                    if pos != "Team" or not first_XI:
                        self.columns_.append(f"{prefix}_{res}")

            if pos != "Team":
                if first_XI:
                    self.columns_.append(f"{where}_{pos}_n_players")
                else:
                    self.columns_.append(f"{where}_{pos}_sub_n_players")

        if self.exact_results:
            self.columns_.append("home_goals")
            self.columns_.append("away_goals")
        else:
            self.columns_.append("result")

        return self

    def good_or_bad(self, team, league, season):
        max_num = 20
        if league == "D1":  # Bundesliga has only 18 teams
            max_num = 18
        pos = table_pos[(team, season)]
        return pos / max_num < 0.5

    def fill_info(self, team, op_team, where, season, date, teams, keepers, players):
        # print("======{team}=======")
        cumulative_team_stat = np.zeros(len(self.team_stats_))
        s = 0
        col = "ht" if where == "home" else "at"
        op_col = "at" if where == "home" else 'ht'

        matches = ginf.loc[(ginf[col] == team) & (ginf["season"] == season) & ginf["adv_stats"]]
        league = matches["league"].values[0]
        gob_team = self.good_or_bad(team, league, season)
        gob_op = self.good_or_bad(op_team, league, season)

        if where == "home":
            player_columns = [f"hp{i}" for i in range(1, 12)] + [f"hs{i}" for i in range(1, 13)]
        else:
            player_columns = [f"ap{i}" for i in range(1, 12)] + [f"as{i}" for i in range(1, 13)]
        lineups_to_extract = []

        team_db = teams.loc[(teams["name"] == team) & (teams["season"] == season)]
        lineup_db = lineups.loc[(lineups[f"{where}_team"] == team) & (lineups["season"] == season)]
        for _, match in matches.iterrows():
            if gob_op == self.good_or_bad(match[op_col], league, season):
                new_date = match["date"]
                cumulative_team_stat += team_db.loc[(team_db["date"] == new_date), self.team_stats_].values.reshape(-1)
                lineups_to_extract.append((lineup_db.loc[lineup_db["date"] == new_date], new_date))
                s += 1
        # print(cumulative_team_stat)

        # first test
        # return cumulative_team_stat, lineups_to_extract

        lineup = lineups.loc[
            (lineups[f"{where}_team"] == team) & (lineups["date"] == date), player_columns].values.reshape(-1)
        lineup = [p for p in lineup if not isnull(p)]
        cpstats = []
        for pl in lineup:
            pos = player_pos[name_to_url[(pl, season, team)]]
            db = keepers if pos == "Goalkeeper" else players
            stat_columns = self.keeper_stat_ if pos == "Goalkeeper" else self.player_stat_

            cpstats.append(np.zeros(len(stat_columns)))
            n_games = 0

            player_db = db.loc[(db["name"] == pl) & (db["team"] == team) & (db["season"] == season)]

            for game, new_date in lineups_to_extract:
                if pl in game.values.reshape(-1):

                    stat_row = player_db.loc[(player_db["date"] == date), stat_columns]
                    try:
                        assert len(stat_row) < 2
                    except AssertionError as e:
                        print(len(stat_row))
                        print(pl, new_date, pos)
                        raise e
                    # nie licze w żaden sposób ile razy ktoś był na ławce rezerwowych
                    if len(stat_row) == 1 and stat_row["time_played"].values[0] > 0:
                        n_games += 1
                        cpstats[-1] += stat_row.values.reshape(-1)

            if n_games > 0:
                cpstats[-1] /= n_games

        cumulative_team_stat /= s

        return cumulative_team_stat, cpstats

    def combine(self, ctstats, cpstats):

        combined = ctstats
        for first_XI, pos in itertools.product([True, False], ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']):
            try:
                combined = np.concatenate((combined, cpstats[(pos, first_XI)]))
            except BaseException as e:
                print(combined)
                print(pos, first_XI)
                print(cpstats[(pos, first_XI)])
                raise e
        return combined

    def res_index(self, res):
        if res == "W":
            return 0
        if res == "X":
            return 1
        if res == "A":
            return 2

    def get_from_history(self, team, season, lineup):

        if self._team_history.get((team, season)) is None:
            return None
        if self.n_games is not None and len(self._team_history[(team, season)]) < self.n_games:
            return None
        if self.n_games is None and len(self._team_history[(team, season)]) == 0:
            return None

        # print(f"=={team}==")

        tstat_res = np.zeros(len(self.team_stats_))
        tstat_form = np.zeros(len(self.team_stats_))
        last_date = None
        res_total, res_form = np.zeros(3), np.zeros(3)
        for ind, (res, date, stat_single) in enumerate(self._team_history[(team, season)]):
            tstat_res += stat_single
            res_total[self.res_index(res)] += 1

            if self.n_games is not None and ind < self.n_games:
                last_date = date
                tstat_form += stat_single
                res_form[self.res_index(res)] += 1

        team_played_total = len(self._team_history[(team, season)])
        tstat_res /= team_played_total
        res_total /= team_played_total
        if self.n_games is not None:
            tstat_form /= self.n_games
            res_form /= self.n_games
            tstat_res = np.concatenate((tstat_res, res_total, tstat_form, res_form))
        else:
            tstat_res = np.concatenate((tstat_res, res_total))

        pstat_res, n_of_players = {}, {}
        pstat_form = {}
        for first_XI, pos in itertools.product([True, False], ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']):
            if pos == "Goalkeeper":
                pstat_res[(pos, first_XI)] = np.zeros(len(self.keeper_stat_) + 3)
                pstat_form[(pos, first_XI)] = np.zeros(len(self.keeper_stat_) + 3)
            else:
                pstat_res[(pos, first_XI)] = np.zeros(len(self.player_stat_) + 3)
                pstat_form[(pos, first_XI)] = np.zeros(len(self.player_stat_) + 3)
            n_of_players[(pos, first_XI)] = 0

        for pl, first_XI in zip(lineup, [True] * 11 + [False] * 12):
            url = name_to_url[(pl, season, team)]
            pos = player_pos[url]

            cpstat_singlepl = np.zeros(len(self.keeper_stat_)) if pos == "Goalkeeper" else np.zeros(
                len(self.player_stat_))
            cpstat_form = np.zeros(len(self.keeper_stat_)) if pos == "Goalkeeper" else np.zeros(len(self.player_stat_))
            res_total, res_form = np.zeros(3), np.zeros(3)

            for res, date, pstat_single in self._player_history.get((url, season), []):
                cpstat_singlepl += pstat_single
                res_total[self.res_index(res)] += 1
                if last_date is not None and date >= last_date:
                    cpstat_form += pstat_single
                    res_form[self.res_index(res)] += 1

            # 12 columns is "time_played" in minutes, we will use it to calculate stats per 90 minutes
            # time_played will be turned into average play time per game
            # last column is games played, we will turn it into fractions of games that the players has appeared in

            n_minutes = cpstat_singlepl[12]
            if n_minutes > 0:
                games_played_stat = cpstat_singlepl[-1]
                cpstat_singlepl = cpstat_singlepl / n_minutes * 90
                cpstat_singlepl[-1] = games_played_stat / team_played_total
                cpstat_singlepl[12] = n_minutes / team_played_total
                res_total /= np.sum(res_total)

            if last_date is not None:
                n_minutes_form = cpstat_form[12]
                if n_minutes_form > 0:
                    games_played_stat = cpstat_form[-1]
                    cpstat_form = cpstat_form / n_minutes_form * 90
                    cpstat_form[-1] = games_played_stat / self.n_games
                    cpstat_form[12] = n_minutes_form / self.n_games
                    res_form /= np.sum(res_form)

            if pos == "Goalkeeper":
                if cpstat_singlepl[5] != 0:
                    raise ValueError
                if (cpstat_singlepl[6] + cpstat_singlepl[7]) > 0:
                    cpstat_singlepl[5] = cpstat_singlepl[6] / (cpstat_singlepl[6] + cpstat_singlepl[7])

                if self.n_games is not None:
                    if (cpstat_form[6] + cpstat_form[7]) > 0:
                        cpstat_form[5] = cpstat_form[6] / (cpstat_form[6] + cpstat_form[7])

            pstat_res[(pos, first_XI)] += np.concatenate((cpstat_singlepl, res_total))
            pstat_form[(pos, first_XI)] += np.concatenate((cpstat_form, res_form))
            n_of_players[(pos, first_XI)] += 1

        # print()
        # print(len(tstat_res))

        for first_XI, pos in itertools.product([True, False], ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']):
            if n_of_players[(pos, first_XI)] > 0:
                pstat_res[(pos, first_XI)] /= n_of_players[(pos, first_XI)]
                pstat_form[(pos, first_XI)]

            if self.n_games is None:
                pstat_res[(pos, first_XI)] = np.concatenate(
                    (pstat_res[(pos, first_XI)], [n_of_players[(pos, first_XI)]]))
            else:
                pstat_res[(pos, first_XI)] = np.concatenate((pstat_res[(pos, first_XI)],
                                                             pstat_form[(pos, first_XI)],
                                                             [n_of_players[(pos, first_XI)]]))
            # print(len(pstat_res[(pos, first_XI)]))

        return self.combine(tstat_res, pstat_res)

    # dodać pozycję ligową z poprzedniego sezonu
    # testing

    def add_to_history(self, team, season, tstats, pstats, lineup, date, result):

        if self._team_history.get((team, season)) is None:
            self._team_history[(team, season)] = []
        self._team_history[(team, season)].insert(0, (result, date, tstats))

        for pl, stat in zip(lineup, pstats):
            url = name_to_url[(pl, season, team)]

            if self._player_history.get((url, season)) is None:
                self._player_history[(url, season)] = []

            self._player_history[(url, season)].insert(0, (result, date, stat))

    def get_info_from_dbs(self, team, season, date, lineup, teams, players, keepers):
        tstats = teams.loc[(teams["name"] == team) & (teams["date"] == date), self.team_stats_].values.reshape(-1)
        pstats = []
        players_db = players.loc[(players["date"] == date) & (players["team"] == team)]
        keepers_db = keepers.loc[(keepers["date"] == date) & (keepers["team"] == team)]
        for pl in lineup:
            pos = player_pos[name_to_url[(pl, season, team)]]
            db = keepers_db if pos == "Goalkeeper" else players_db
            stat_columns = self.keeper_stat_ if pos == "Goalkeeper" else self.player_stat_

            # pstat_single = db.loc[(db["name"] == pl) & (db["date"] == date) & (db["team"] == team), stat_columns].values.reshape(-1)
            pstat_single = db.loc[db["name"] == pl, stat_columns].values.reshape(-1)

            if len(pstat_single) > 0:
                pstats.append(pstat_single)
            else:
                pstats.append(np.zeros(len(stat_columns)))

        return tstats, pstats

    def get_result(self, hgoals, agoals):
        if self.exact_results:
            return [hgoals, agoals]
        if hgoals > agoals:
            return ["H"]
        if hgoals == agoals:
            return ["X"]
        return ["A"]

    def transform(self, X, y=None):
        teams, players, keepers = X

        X = pd.DataFrame(columns=self.columns_)

        # for _, row in ginf.iterrows():
        for _, row in tqdm(ginf.iterrows(), total=len(ginf)):
            season, date, ht, at = row["season"], row["date"], row["ht"], row["at"]
            hgoals, agoals = row["fthg"], row["ftag"]

            # print(f"{ht:25} - {at:25}")
            try:

                lineup = lineups.loc[
                    (lineups["season"] == season) & (lineups["home_team"] == ht) & (lineups["away_team"] == at)]
                home_lineup = lineup.loc[:,
                              [f"hp{i}" for i in range(1, 12)] + [f"hs{i}" for i in range(1, 13)]].values.reshape(-1)
                home_lineup = [hp for hp in home_lineup if not isnull(hp)]
                away_lineup = lineup.loc[:,
                              [f"ap{i}" for i in range(1, 12)] + [f"as{i}" for i in range(1, 13)]].values.reshape(-1)
                away_lineup = [ap for ap in away_lineup if not isnull(ap)]

                homestats = self.get_from_history(ht, season, home_lineup)
                awaystats = self.get_from_history(at, season, away_lineup)
                result = self.get_result(hgoals, agoals)
                if result == ["H"]:
                    homeres, awayres = "W", "L"
                elif result == ["A"]:
                    homeres, awayres = "L", "W"
                elif result == ["X"]:
                    homeres, awayres = "X", "X"

                if not row["adv_stats"]:
                    htstats, hpstats = self.fill_info(ht, at, "home", season, date, teams, keepers, players)
                    atstats, apstats = self.fill_info(at, ht, "away", season, date, teams, keepers, players)

                else:
                    htstats, hpstats = self.get_info_from_dbs(ht, season, date, home_lineup, teams, players, keepers)
                    atstats, apstats = self.get_info_from_dbs(at, season, date, away_lineup, teams, players, keepers)

                self.add_to_history(ht, season, htstats, hpstats, home_lineup, date, homeres)
                self.add_to_history(at, season, atstats, apstats, away_lineup, date, awayres)

                if homestats is not None and awaystats is not None:
                    # print(len(homestats), len(awaystats), len(result), len(self.columns_))
                    X.loc[len(X)] = np.concatenate((homestats, awaystats, result))  # results, i inne dodatkowe kolumny

            except BaseException as e:
                print()
                print(f"{ht} - {at} ({season})")
                # temp_sol = X
                # ind = ind_curr
                raise e

        return X


dbuilder = DatabaseBuilder(n_games=5)
dbuilder.fit((teams, players, goalkeepers))
X = dbuilder.transform((teams, players, goalkeepers))
X.to_csv("X.csv", index=False)
