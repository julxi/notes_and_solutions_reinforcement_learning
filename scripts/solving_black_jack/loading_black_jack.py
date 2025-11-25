from pathlib import Path
import pickle
import environment.black_jack as black_jack


# or quick one-liner without helper:
with open("results/blackjack_20000000_5.pkl", "rb") as f:
    package = pickle.load(f)

π = package["pi"]


def str_from_action(action):
    if action == black_jack.Action.HIT:
        return "H"
    elif action == black_jack.Action.STICK:
        return "S"


print("player has ace")

for player_total in range(21, 11, -1):
    print(
        player_total,
        " ",
        " ".join(
            [
                str_from_action(π[black_jack.State(player_total, True, showing_card)])
                for showing_card in range(2, 12)
            ]
        ),
    )
print("     " + " ".join([str(i) for i in range(2, 12)]))


print("player has no ace")
for player_total in range(21, 11, -1):
    print(
        player_total,
        " ",
        " ".join(
            [
                str_from_action(π[black_jack.State(player_total, False, showing_card)])
                for showing_card in range(2, 12)
            ]
        ),
    )
print("     " + " ".join([str(i) for i in range(2, 12)]))
