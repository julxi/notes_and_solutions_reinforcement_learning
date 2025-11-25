import pytest
import random
from enum import Enum
from environment.black_jack import BlackJack, State, Action


def test_generate_state_space():
    rng = random.Random(3)
    env = BlackJack(rng)

    state_set = set()
    for _ in range(9_000):
        state, _, done = env.reset()

        while not done:
            state_set.add(state)
            state, _, done = env.step(Action.HIT)

    for state in env.state_space:
        assert state in state_set
        state_set.remove(state)

    assert len(state_set) == 0


def test_reset_natural_player_only(monkeypatch):
    card_values = [11, 10, 2, 3]

    def mock_draw_card(self):
        return card_values.pop(0)

    monkeypatch.setattr(BlackJack, "_draw_card", mock_draw_card)
    env = BlackJack()
    state, reward, done = env.reset()
    assert reward == 1
    assert done
    assert state.dealer_revealed == 5


def test_reset_natural_dealer_only(monkeypatch):
    card_values = [2, 3, 11, 10]

    def mock_draw_card(self):
        return card_values.pop(0)

    monkeypatch.setattr(BlackJack, "_draw_card", mock_draw_card)
    env = BlackJack()
    state, reward, done = env.reset()
    assert reward == -1
    assert done
    assert state.dealer_revealed == 21


def test_reset_natural_both(monkeypatch):
    card_values = [11, 10, 11, 10]

    def mock_draw_card(self):
        return card_values.pop(0)

    monkeypatch.setattr(BlackJack, "_draw_card", mock_draw_card)
    env = BlackJack()
    state, reward, done = env.reset()
    assert reward == 0
    assert done


def test_reset_force_hit(monkeypatch):
    card_values = [5, 5, 2, 3, 6]

    def mock_draw_card(self):
        return card_values.pop(0)

    monkeypatch.setattr(BlackJack, "_draw_card", mock_draw_card)
    env = BlackJack()
    state, reward, done = env.reset()
    assert state.player_total == 16
    assert reward == 0
    assert not done


def test_set_state_never_gives_dealer_21():
    env = BlackJack()
    for _ in range(10):
        for state in env.state_space:
            env.set_state(state)
            assert env._dealer_total != 21
            assert env._player_total == state.player_total
            assert env._player_soft == state.is_soft


def test_hit_hard_21_always_bust():
    env = BlackJack()
    for _ in range(1000):
        env.set_state(State(21, False, 10))
        state, reward, done = env.step(Action.HIT)
        assert reward == -1
        assert done == True


def test_add_card_below_21_keeps_soft():
    env = BlackJack()
    total, is_soft = env._add_card_to_hand(15, False, 5)
    assert total == 20
    assert is_soft == False


def test_add_ace_over_21_reduces_ace():
    env = BlackJack()
    total, is_soft = env._add_card_to_hand(15, False, 11)
    assert total == 16
    assert is_soft == False


def test_add_ace_to_soft_21_does_not_bust():
    env = BlackJack()
    total, is_soft = env._add_card_to_hand(21, True, 11)
    assert total == 12
    assert is_soft == False


def test_step_hit_bust(monkeypatch):
    env = BlackJack()
    env._player_total = 20
    env._player_soft = False
    env._dealer_upcard = 10
    monkeypatch.setattr(env, "_draw_card", lambda: 2)
    state, reward, done = env.step(Action.HIT)
    assert reward == -1
    assert done
    assert state.player_total == 22


def test_step_hit_safe(monkeypatch):
    env = BlackJack()
    env._player_total = 16
    env._player_soft = False
    env._dealer_upcard = 10
    monkeypatch.setattr(env, "_draw_card", lambda: 3)
    state, reward, done = env.step(Action.HIT)
    assert reward == 0
    assert not done
    assert state.player_total == 19


def test_step_stick_dealer_bust(monkeypatch):
    env = BlackJack()
    env._player_total = 18
    env._dealer_total = 16
    env._dealer_soft = False
    env._player_soft = False
    env._dealer_upcard = 7
    monkeypatch.setattr(env, "_draw_card", lambda: 6)
    state, reward, done = env.step(Action.STICK)
    assert reward == 1
    assert done
    assert state.dealer_revealed == 22


def test_step_stick_dealer_wins(monkeypatch):
    env = BlackJack()
    env._player_total = 18
    env._dealer_total = 16
    env._dealer_upcard = 10
    env._dealer_soft = False
    env._player_soft = False
    monkeypatch.setattr(env, "_draw_card", lambda: 3)
    state, reward, done = env.step(Action.STICK)
    assert reward == -1
    assert state.dealer_revealed == 19


def test_step_stick_tie():
    env = BlackJack()
    env._player_total = 17
    env._dealer_total = 17
    env._dealer_upcard = 9
    env._dealer_soft = False
    env._player_soft = False
    state, reward, done = env.step(Action.STICK)
    assert reward == 0
    assert state.dealer_revealed == 17


def test_hit_card_soft_to_hard(monkeypatch):
    env = BlackJack()
    monkeypatch.setattr(env, "_draw_card", lambda: 5)
    total, is_soft = env._hit_card(17, True)
    assert total == 12
    assert not is_soft


def test_check_natural_both():
    env = BlackJack()
    payout, is_natural = env._check_natural(21, 21)
    assert payout == 0
    assert is_natural


def test_check_natural_player_only():
    env = BlackJack()
    payout, is_natural = env._check_natural(21, 20)
    assert payout == 1
    assert is_natural


def test_check_natural_dealer_only():
    env = BlackJack()
    payout, is_natural = env._check_natural(19, 21)
    assert payout == -1
    assert is_natural


def test_check_natural_neither():
    env = BlackJack()
    payout, is_natural = env._check_natural(20, 20)
    assert payout == 0
    assert not is_natural
