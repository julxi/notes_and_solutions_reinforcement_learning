import random
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

Reward = int


class Action(Enum):
    HIT = auto()
    STICK = auto()


@dataclass(frozen=True)
class State:
    """State visible to player"""

    player_total: int
    is_soft: bool
    dealer_revealed: int


class BlackJack:
    """
    Simplified Blackjack environment
    with infinite deck assumption.
    """

    ACE_HIGH_VALUE = 11
    FACE_VALUE = 10
    TWENTY_ONE = 21
    DEALER_STICK = 17
    PLAYER_MIN = 12

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()
        self.state_space = [
            State(player_total, player_soft, dealer_up_card)
            for player_total in range(self.PLAYER_MIN, self.TWENTY_ONE + 1)
            for player_soft in (True, False)
            for dealer_up_card in range(2, self.ACE_HIGH_VALUE + 1)
        ]
        self.action_space = [Action.HIT, Action.STICK]

    def reset(self) -> Tuple[State, Reward, bool]:
        """
        Deal initial hands to player and dealer.
        Then returns state, reward, terminated.
        """
        # Deal two cards to player and dealer
        p_total, p_soft, _ = self._deal_hand()
        d_total, d_soft, d_up = self._deal_hand()

        # Check for naturals
        reward, done = self._check_natural(p_total, d_total)
        if done:
            return State(p_total, p_soft, d_total), reward, done

        # Force-hit until safe minimum (player can't bust below 12)
        while p_total < self.PLAYER_MIN:
            p_total, p_soft = self._hit_card(p_total, p_soft)

        # Save state
        self._player_total, self._player_soft = p_total, p_soft
        self._dealer_total, self._dealer_soft = d_total, d_soft
        self._dealer_upcard = d_up

        return State(p_total, p_soft, d_up), 0, False

    def set_state(self, state: State):
        # sets non-terminal state

        # player state
        self._player_total = state.player_total
        self._player_soft = state.is_soft
        self._dealer_upcard = state.dealer_revealed

        # base dealer state
        base_total = state.dealer_revealed
        base_soft = state.dealer_revealed == self.ACE_HIGH_VALUE

        # draw hidden card repeatedly until dealer does not have blackjack
        dt, ds = self._hit_card(base_total, base_soft)
        while dt == self.TWENTY_ONE:
            dt, ds = self._hit_card(base_total, base_soft)

        # dealer state
        self._dealer_total = dt
        self._dealer_soft = ds

        return 0

    def step(self, action: Action) -> Tuple[State, Reward, bool]:
        """
        Returns state, reward, terminated after taken action.
        """
        return self._step_hit() if action is Action.HIT else self._step_stick()

    def _step_hit(self) -> Tuple[State, int, bool]:
        pt, ps = self._hit_card(self._player_total, self._player_soft)
        self._player_total, self._player_soft = pt, ps
        done = pt > self.TWENTY_ONE
        reward = -1 if done else 0
        return State(pt, ps, self._dealer_upcard), reward, done

    def _step_stick(self) -> Tuple[State, int, bool]:
        # Dealer draws until stick threshold
        dt, ds = self._dealer_total, self._dealer_soft
        while dt < self.DEALER_STICK:
            dt, ds = self._hit_card(dt, ds)

        pt = self._player_total
        reward = 1 if dt > self.TWENTY_ONE or pt > dt else -1 if dt > pt else 0

        return State(pt, self._player_soft, dt), reward, True

    def _deal_hand(self) -> Tuple[int, bool, int]:
        """Deal two initial cards and return (total, is_soft, first_card)."""
        first_card, is_soft = self._hit_card(0, False)
        total, is_soft = self._hit_card(first_card, is_soft)

        return total, is_soft, first_card

    def _hit_card(self, total: int, is_soft: bool) -> Tuple[int, bool]:
        """Add card do hand"""
        card = self._draw_card()
        return self._add_card_to_hand(total, is_soft, card)

    def _add_card_to_hand(
        self, total: int, is_soft: bool, card: int
    ) -> Tuple[int, bool]:
        total += card
        is_ace = card == self.ACE_HIGH_VALUE

        usable_aces = (1 if is_soft else 0) + (1 if is_ace else 0)

        while total > self.TWENTY_ONE and usable_aces > 0:
            total -= 10
            usable_aces -= 1

        is_soft = usable_aces > 0
        return total, is_soft

    def _draw_card(self) -> int:
        """Draw a card: numbered cards, face cards as 10, Ace as 11."""
        r = self.rng.randint(1, 13)
        return self.ACE_HIGH_VALUE if r == 1 else self.FACE_VALUE if r > 10 else r

    def _check_natural(self, player_total: int, dealer_total: int) -> Tuple[int, bool]:
        """Return (payout, is_natural) for Blackjack naturals."""
        player_nat = player_total == self.TWENTY_ONE
        dealer_nat = dealer_total == self.TWENTY_ONE
        is_natural = player_nat or dealer_nat
        payout = int(player_nat) - int(dealer_nat)
        return payout, is_natural
