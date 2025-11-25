from ..environment.black_jack import State, Action


def print_policy(π):
    def str_from_action(action):
        if action == Action.HIT:
            return "H"
        else:
            return "S"

    print("player has ace")
    for player_total in range(21, 11, -1):
        print(
            player_total,
            " ",
            " ".join(
                [
                    str_from_action(π[State(player_total, True, showing_card)])
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
                    str_from_action(π[State(player_total, False, showing_card)])
                    for showing_card in range(2, 12)
                ]
            ),
        )
    print("     " + " ".join([str(i) for i in range(2, 12)]))
