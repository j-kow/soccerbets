from bs4 import BeautifulSoup
from time import sleep
from unidecode import unidecode
import numpy as np
import pandas as pd
import re
import pickle

from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

wd = webdriver.Chrome('chromedriver', options=chrome_options)

SITE = "https://int.soccerway.com"
pos_translator = {"Goalkeeper": 1, "Bramkarz": 1,
                  "Defender": 2, "Obrońca": 2,
                  "Midfielder": 3, "Pomocnik": 3,
                  "Attacker": 4, "Napastnik": 4}
soccerway_to_kaggle_team_names = {
    'Wigan Athletic': 'Wigan',
    'West Ham United': 'West Ham',
    'West Bromwich Albion': 'West Brom',
    'Wolfsburg': 'VfL Wolfsburg',
    'Stuttgart': 'VfB Stuttgart',
    'Pescara': 'US Pescara',
    'Eintracht Braunschweig': 'TSV Eintracht Braunschweig',
    'Swansea City': 'Swansea',
    'Rennes': 'Stade Rennes',
    'Saint-Etienne': 'St Etienne',
    'Darmstadt 98': 'SV Darmstadt 98',
    'Paderborn': 'SC Paderborn',
    'Freiburg': 'SC Freiburg',
    'Köln': 'FC Cologne',
    'Hamburger SV': 'Hamburg SV',
    'Monaco': 'AS Monaco',
    'Bolton Wanderers': 'Bolton',
    'Angers SCO': 'Angers',
    'Augsburg': 'FC Augsburg',
    'Hoffenheim': 'TSG Hoffenheim',
    'Tottenham Hotspur': 'Tottenham',
    'Sporting Gijón': 'Sporting Gijon',
    'Dijon': 'Dijon FCO',
    'Wolverhampton Wanderers': 'Wolves',
    'Reims': 'Stade de Reims',
    'Greuther Fürth': 'SpVgg Greuther Furth',
    'Evian TG': 'Evian Thonon Gaillard',
    'Ajaccio': 'AC Ajaccio',
    'Milan': 'AC Milan',
    'Auxerre': 'AJ Auxerre',
    'Nancy': 'AS Nancy Lorraine',
    'Roma': 'AS Roma',
    'Deportivo Alavés': 'Alaves',
    'Almería': 'Almeria',
    'Athletic Club': 'Athletic Bilbao',
    'Blackburn Rovers': 'Blackburn',
    "Borussia M'gladbach": 'Borussia Monchengladbach',
    'AFC Bournemouth': 'Bournemouth',
    'Cardiff City': 'Cardiff',
    'Chievo': 'Chievo Verona',
    'Córdoba': 'Cordoba',
    'Deportivo La Coruña': 'Deportivo La Coruna',
    'Ingolstadt': 'FC Ingolstadt 04',
    'Gazélec Ajaccio': 'GFC Ajaccio',
    'Hertha BSC': 'Hertha Berlin',
    'Hull City': 'Hull',
    # '': 'Karlsruher SC',
    'Olympique Lyonnais': 'Lyon',
    'Mainz 05': 'Mainz',
    'Málaga': 'Malaga',
    'Manchester United': 'Manchester Utd',
    'Olympique Marseille': 'Marseille',
    'Newcastle United': 'Newcastle',
    'Nürnberg': 'Nurnberg',
    'PSG': 'Paris Saint-Germain',
    'Queens Park Rangers': 'QPR',

    "Borussia M'gla…": "Borussia Monchengladbach",
    "West Bromwich …": "West Brom",
    "Queens Park Ra…": "QPR",
    "Wolverhampton …": "Wolves",
    "Deportivo La C…": "Deportivo La Coruna",
    "Olympique Mars…": "Marseille",
    "Eintracht Fran…": "Eintracht Frankfurt",
    "Eintracht Brau…": "TSV Eintracht Braunschweig"
}


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_save(path, ob):
    with open(path, "wb") as f:
        pickle.dump(ob, f)


class Cache:
    def __init__(self, drive_root=""):
        self.player_stats = {}
        self.player_pos = {}
        self.url_to_name = {}
        self.name_to_url = {}
        self.missing_data = []
        self.visited = set()
        self.table_pos = {}

        self.drive_root = drive_root

    def save_cache(self):
        pickle_save(self.drive_root + "player_stats.pkl", self.player_stats)
        pickle_save(self.drive_root + "player_pos.pkl", self.player_pos)
        pickle_save(self.drive_root + "url_to_name.pkl", self.url_to_name)
        pickle_save(self.drive_root + "name_to_url.pkl", self.name_to_url)
        pickle_save(self.drive_root + "visited.pkl", self.visited)
        pickle_save(self.drive_root + "missing_data.pkl", self.missing_data)
        pickle_save(self.drive_root + "table_pos.pkl", self.table_pos)

    def save_backup(self):
        pickle_save(self.drive_root + "player_stats_bkp.pkl", self.player_stats)
        pickle_save(self.drive_root + "player_pos_bkp.pkl", self.player_pos)
        pickle_save(self.drive_root + "url_to_name_bkp.pkl", self.url_to_name)
        pickle_save(self.drive_root + "name_to_url_bkp.pkl", self.name_to_url)
        pickle_save(self.drive_root + "visited_bkp.pkl", self.visited)
        pickle_save(self.drive_root + "missing_data_bkp.pkl", self.missing_data)
        pickle_save(self.drive_root + "table_pos_bkp.pkl", self.table_pos)

    @staticmethod
    def load_cache(drive_root=""):
        c = Cache(drive_root)

        c.player_stats = pickle_load(c.drive_root + "player_stats.pkl")
        c.player_pos = pickle_load(c.drive_root + "player_pos.pkl")
        c.url_to_name = pickle_load(c.drive_root + "url_to_name.pkl")
        c.name_to_url = pickle_load(c.drive_root + "name_to_url.pkl")
        c.visited = pickle_load(c.drive_root + "visited.pkl")
        c.missing_data = pickle_load(c.drive_root + "missing_data.pkl")
        c.table_pos = pickle_load(c.drive_root + "table_pos.pkl")

        return c


def get_data(link, verbose=True, delay=10):
    if verbose:
        print(f"visiting {link}")

    wd.get(link)
    sleep(delay)
    soup = BeautifulSoup(wd.page_source, 'html.parser')
    return soup


def get_player_data(cache, url, season, team, with_pos=False, verbose=True):
    if url == "/players/roel-brouwers/16901/":  # roel brouwers has two urls
        if with_pos:
            return pos_translator["Defender"], "roel brouwers"
        return "roel brouwers"

    if url not in cache.visited:
        player_soup = get_data(SITE + url, verbose=verbose)

        try:
            weight = int(player_soup.find("dd", {"data-weight": "weight"}).text.split()[0])
        except AttributeError:
            weight = np.nan
        try:
            height = int(player_soup.find("dd", {"data-height": "height"}).text.split()[0])
        except AttributeError:
            height = np.nan
        try:
            pos = player_soup.find("dd", {"data-position": "position"}).text
        except AttributeError:
            pos = "Defender"

        name = unidecode(re.search(r"(?<= - ).+(?= - Profil)", wd.title).group(0)).lower()
        if len(name.split()[0]) == 2 and name.split()[0][1] == ".":

            first_name = None
            first_names = unidecode(player_soup.find("dd", {"data-first_name": "first_name"}).text).lower().split()
            matching_letter = name.split()[0][0].lower()

            for poss_name in first_names:
                if poss_name[0] == matching_letter and first_name is None:
                    first_name = poss_name

            if first_name is None:
                first_name = first_names[0]

            surname = unidecode(player_soup.find("dd", {"data-last_name": "last_name"}).text).lower()
            name = f"{first_name} {surname}".lower()
        else:
            first_names = unidecode(player_soup.find("dd", {"data-first_name": "first_name"}).text).lower().split()
            surnames = unidecode(player_soup.find("dd", {"data-last_name": "last_name"}).text).lower().split()

            if name == surnames[0]:
                name = first_names[0] + " " + name

        cache.url_to_name[url] = name

        assert cache.name_to_url.get((name, season, team)) is None
        cache.name_to_url[(name, season, team)] = url

        cache.player_stats[url] = [weight, height]
        cache.player_pos[url] = pos
        cache.visited.add(url)

    name = cache.url_to_name[url]
    if cache.name_to_url.get((name, season, team)) is None:
        cache.name_to_url[(name, season, team)] = url

    if with_pos:
        return pos_translator[cache.player_pos[url]], name
    return name


def failed_load(cache, home_team, away_team, season, date, lineups, verbose=True):
    cache.missing_data.append((home_team, away_team, season))
    row = [season, date, home_team] + [np.NaN] * 23 + [away_team] + [np.NaN] * 23
    if verbose:
        print(f"COULD NOT FIND INFO ABOUT: {home_team} - {away_team} ({season})")
    lineups.loc[len(lineups)] = row
    return row


def get_lineup(cache, home_team, away_team, season, match_soup, lineups, verbose=True):
    day, month, year = match_soup.find("div", {"class": "details"}).a.text.split("/")
    date = f"{year}-{month}-{day}"

    try:
        home_links = match_soup.find_all("div", {"class": "combined-lineups-container"})[0].find("div", {
            "class": "container left"}).find_all("a", href=True)
        away_links = match_soup.find_all("div", {"class": "combined-lineups-container"})[0].find("div", {
            "class": "container right"}).find_all("a", href=True)
    except IndexError:  # No lineup available
        return failed_load(cache, home_team, away_team, season, date, lineups, verbose=verbose)
    if len(home_links) != 12 or len(away_links) != 12:
        return failed_load(cache, home_team, away_team, season, date, lineups, verbose=verbose)

    home_subs = match_soup.find_all("div", {"class": "combined-lineups-container"})[1].find("div", {
        "class": "container left"}).find_all("a", href=True, class_=True)
    away_subs = match_soup.find_all("div", {"class": "combined-lineups-container"})[1].find("div", {
        "class": "container right"}).find_all("a", href=True, class_=True)
    if len(home_subs) > 12 or len(away_subs) > 12:
        return failed_load(cache, home_team, away_team, season, date, lineups, verbose=verbose)

    home_subs_with_pos = sorted(
        [get_player_data(cache, link['href'], season, home_team, True, verbose) for link in home_subs])
    away_subs_with_pos = sorted(
        [get_player_data(cache, link['href'], season, away_team, True, verbose) for link in away_subs])

    home_players = [get_player_data(cache, link['href'], season, home_team, verbose=verbose) for link in
                    home_links[:-1]] + [sub for _, sub in home_subs_with_pos]
    away_players = [get_player_data(cache, link['href'], season, away_team, verbose=verbose) for link in
                    away_links[:-1]] + [sub for _, sub in away_subs_with_pos]

    row = [season, date, home_team] + home_players + [np.NaN] * (23 - len(home_players)) + [
        away_team] + away_players + [np.NaN] * (23 - len(away_players))
    lineups.loc[len(lineups)] = row
    return row


def all_games_from_season(cache, season_url, season, lineups, verbose=True):
    global wd
    season_soup = get_data(season_url, verbose=verbose)

    for test in season_soup.find("table", {
        "id": "page_competition_1_block_competition_tables_13_block_competition_league_table_1_table"}).tbody.find_all(
        "tr"):
        rank = int(test.find("td", class_=re.compile("rank")).text)
        name = test.find("a", href=True).text
        name = soccerway_to_kaggle_team_names.get(name, name)
        if re.search("…", name) is not None:
            print(name)
            raise KeyError
        print(f"added {name} : {rank}")
        cache.table_pos[(name, season)] = rank

    rounds = wd.find_element_by_xpath(
        '//*[@id="page_competition_1_block_competition_matches_summary_11_page_dropdown"]')
    round_max = int(rounds.find_elements_by_tag_name("option")[-1].text)
    links = []
    for round in range(round_max, 0, -1):
        # for option in rounds.find_elements_by_tag_name('option'):
        #         option.click()
        #         wd.execute_script("arguments[0].click();", option)
        #         sleep(5)
        round_soup = BeautifulSoup(wd.page_source, "html.parser")
        #         round = option.text

        for link in round_soup.find("table", {"class": "matches"}).find_all("a", href=re.compile("matches")):
            if link.text != "View events" and link.text != "Zobacz wydarzenia":
                # home_team, away_team = re.search(f"(?<={league_code}/)[\w-]+/[\w-]+/", link['href']).group(0).split("/")[:2]
                home_team, away_team = [team.text.strip() for team in
                                        link.parent.parent.find_all("a", href=re.compile("teams"))]
                home_team = soccerway_to_kaggle_team_names.get(home_team, home_team)
                away_team = soccerway_to_kaggle_team_names.get(away_team, away_team)

                # if not visited
                if len(lineups.loc[(lineups['season'] == season) & (lineups['home_team'] == home_team) & (
                        lineups['away_team'] == away_team)]) == 0:
                    links.append((SITE + link['href'], home_team, away_team, round))

        prev_elem = wd.find_element_by_xpath(
            '//*[@id="page_competition_1_block_competition_matches_summary_11_previous"]')
        wd.execute_script("arguments[0].click();", prev_elem)
        sleep(6)

    visited_rounds = set()
    for link, ht, at, round in links:
        if verbose:
            if round not in visited_rounds:
                visited_rounds.add(round)
                print(f"===== ROUND {round} =====")
            print(f"{ht} - {at}")
        if (ht, at, season) not in cache.missing_data:
            match_soup = get_data(link, verbose=True)
            get_lineup(cache, ht, at, season, match_soup, lineups, verbose=True)


def all_seasons_of_league(cache, league_soup, lineups, verbose=True, save=True):
    start_year, end_year = 2011, 2017
    for season in range(start_year, end_year + 1):
        season_text = f"{season - 1}/{season}"
        link = league_soup.find("div", {"id": "page_competition_1_block_competition_archive_11"}) \
            .find("a", text=season_text)

        if verbose:
            print(f"========== {season_text} ==========")

        all_games_from_season(cache, SITE + link['href'], season, lineups, True)

        if save:
            cache.save_cache()
            lineups.to_csv("lineups.csv", index=False)


leagues = [
    ("Serie A", "https://int.soccerway.com/national/italy/serie-a/c13/archive/", "serie-a"),
    ("Premier League", "https://int.soccerway.com/national/england/premier-league/c8/archive/", "premier-league"),
    ("La Liga", "https://int.soccerway.com/national/spain/primera-division/c7/archive/", "primera-division"),
    ("Ligue 1", "https://int.soccerway.com/national/france/ligue-1/c16/archive/", "ligue-1"),
    ("Bundesliga", "https://int.soccerway.com/national/germany/bundesliga/c9/archive/", "bundesliga")]


def crawl(verbose=True, start_new=False, save=True):
    if start_new:
        cache = Cache()
        player_names = [f"{first_xi}{no}" for first_xi in ["p", "s"] for no in range(1, 12)] + ["s12"]
        lineups = pd.DataFrame(columns=["season", "date", "home_team"] + [f"h{player}" for player in player_names] +
                                       ["away_team"] + [f"a{player}" for player in player_names])
    else:
        cache = Cache.load_cache()
        lineups = pd.read_csv("../lineups.csv")

    try:
        for league_name, league_url, league_code in leagues:
            if verbose:
                print(f"=============== {league_name} ===============")
            league_soup = get_data(league_url, verbose=verbose)
            all_seasons_of_league(cache, league_soup, lineups, verbose=verbose, save=save)
        return

    except BaseException as e:
        # cache.save_cache()
        # lineups.to_pickle(DRIVE_ROOT + "lineups.pkl")
        print(wd.current_url)
        raise e


crawl(start_new=False, save=False)
