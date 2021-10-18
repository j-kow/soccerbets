import pandas as pd
import itertools


def combine_odds():
    path = "odds/"
    leagues = ["E", "G", "F", "I", "S"]
    seasons = ["17", "16", "15", "14", "13", "12"]
    files = [(i, j) for i, j in itertools.product(leagues, seasons)]

    odds = pd.read_csv("/work/Odds/E17.csv")
    for f in files:
        if f[0] == "E" and f[1] == "17":
            continue

        f_path = path + f[0] + f[1] + ".csv"
        temp = pd.read_csv(f_path)
        odds = pd.concat([odds, temp])

    odds.to_csv("./all_odds.csv", index=False)
