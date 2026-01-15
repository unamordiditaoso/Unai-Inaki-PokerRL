import random
import logging
import numpy as np
from gymnasium import Env, spaces
from treys import Card, Deck, Evaluator
from PokerEquity import estimate_equity

# ======================
# Políticas de oponentes
# ======================

def cards_int_to_str(cards_int):
    return [Card.int_to_str(c) for c in cards_int]

def policy_player1(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return 2 if eq > 0.20 else 0

def policy_player2(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([2,3]) if eq > 0.30 else 0

def policy_player3(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([0,2]) if eq > 0.20 else 0

def policy_player4(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    if eq > 0.15:
        return 2
    else:
        return 0 if random.random() < 0.7 else 2

# ======================
# Entorno Gymnasium
# ======================

class Poker5EnvFull(Env):
    ACTIONS = ["Fold", "Check", "Call", "Bet", "Raise"]
    HAND_RANK_REWARD = {
        9: -0.4,   # High Card
        8:  0.0,   # Pair
        7:  0.2,   # Two Pair
        6:  0.6,   # Trips
        5:  1.2,   # Straight
        4:  1.5,   # Flush
        3:  2,   # Full House
        2:  2.5,   # Quads
        1:  5.0,   # Straight Flush
        0:  6.0    # Royal 
    }

    def _log(self, msg):
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def __init__(self, opponent_policies=None, model_player1=None, model_player2=None, model_player3=None, model_player4=None, starting_stack=1000, small_blind=10, big_blind=20):
        super().__init__()
        self.log_path = "env.log"

        self.model_player1 = model_player1
        self.model_player2 = model_player2
        self.model_player3 = model_player3
        self.model_player4 = model_player4
        self.opponent_policies = opponent_policies
        self.dealer_pos = 0
        self.num_players = 5
        self.agent_id = 0
        self.agent_folded = False
        self.starting_stack = starting_stack
        self.preflop_stack = starting_stack
        self.winners = []
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.observation_space = spaces.Dict({
            "hero_hand": spaces.MultiDiscrete([52, 52]),
            "board": spaces.MultiDiscrete([52+1]*5),
            "stacks": spaces.Box(low=0, high=np.inf, shape=(self.num_players,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "current_bet": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "active_players": spaces.MultiBinary(self.num_players),
            "hand_rank": spaces.Discrete(10)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        self.deck = Deck()
        self.deck.shuffle()
        self.evaluator = Evaluator()
        self.board = []
        
        
        self.stacks = [self.starting_stack]*self.num_players
        self.all_in = [False] * self.num_players
        self.pot = 0
        self.current_bet = self.big_blind
        self.bets = [0]*self.num_players
        self.hands = [self.deck.draw(2) for _ in range(self.num_players)]
        self.active_players = [True]*self.num_players
        self.agent_folded = False
        self.dealer_pos = 0

        return self._get_obs(), {}

    def partial_reset(self):
        self.hands = [None] * self.num_players
        self.board = []
        self.deck.shuffle()

        for i in range(self.num_players):
            if self.stacks[i] <= 0:
                self.stacks[i] = self.starting_stack

        self.hands = [
            sorted(self.deck.draw(2), key=lambda c: Card.get_rank_int(c), reverse=True)
            for _ in range(self.num_players)
        ]

        print("Hero Hand:")
        Card.print_pretty_cards(self.hands[self.agent_id])

        self.active_players = [True] * self.num_players
        self.winners = []

        self.bets = [0] * self.num_players
        self.all_in = [False] * self.num_players
        self.all_check = [False] * self.num_players
        self.current_bet = self.big_blind + self.small_blind
        self.pot = 0

        self.round_stage = 'preflop'
        self.preflop_stack = self.stacks[0]
        self.agent_folded = False

        sb_pos = (self.dealer_pos + 1) % self.num_players
        bb_pos = (self.dealer_pos + 2) % self.num_players

        self.stacks[sb_pos] -= self.small_blind
        self.stacks[bb_pos] -= self.big_blind
        self.bets[sb_pos] = self.small_blind
        self.bets[bb_pos] = self.big_blind
        self.pot = self.small_blind + self.big_blind

        self.dealer_pos = (self.dealer_pos + 1) % self.num_players

        self.reward = 0
        self.done = False
        
        self._log("========================================")
        self._log(f"NEW_HAND | dealer={self.dealer_pos}")
        self._log(f"STACKS_START | {self.stacks}")
        self._log(f"HERO_HAND | {cards_int_to_str(self.hands[self.agent_id])}")

        # Políticas oponentes
        # if self.opponent_policies is None:
        #    self.opponent_policies = [
        #        lambda obs: policy_player1(self.hands[1], obs["board"], num_opponents=4)
        #    ]
        # else:
        #    assert len(self.opponent_policies) == 1

        # Retornar la observación inicial de la mano
        return self._get_obs()

    
    def card_as_index(self, card_int: int) -> int:
        """
        Convierte un entero de Treys a un índice 0..51
        """
        rank = Card.get_rank_int(card_int)  # 0..12
        suit = Card.get_suit_int(card_int)  # 1,2,4,8
        suit_index = {1:0, 2:1, 4:2, 8:3}[suit]
        return rank + 13 * suit_index  # 0..51

    def get_hand_rank(self, hero_hand, board):
        hand = hero_hand
        board_cards = [c for c in board if c != 52]

        if len(board_cards) < 3:
            return 9

        score = self.evaluator.evaluate(hand, board_cards)
        return self.evaluator.get_rank_class(score)

    def preflop_hand_strength(self, hero_hand):
        ranks = sorted([c % 13 for c in hero_hand])
        suited = hero_hand[0] // 13 == hero_hand[1] // 13

        if ranks[0] == ranks[1]:
            if max(ranks) >= 6:
                return 1.0
            else:
                return 0.6
        if suited and max(ranks) >= 8:
            if ranks[0] == ranks[1] + 1:
                return 0.7
            else:
                return 0.4
        if max(ranks) >= 10:
            return 0.5
        if ranks[0] == ranks[1] + 1:
            return 0.1
        return -0.2

    def _get_obs(self):
        board = self.board[:5]
        board += [None]*(5-len(board))
        return {
            "hero_hand": [self.card_as_index(c) for c in self.hands[self.agent_id]],
            "board": [self.card_as_index(c) if c is not None else 52 for c in board],
            "stacks": np.array(self.stacks, dtype=np.float32),
            "pot": np.array([self.pot], dtype=np.float32),
            "current_bet": np.array([self.current_bet], dtype=np.float32),
            "active_players": np.array(self.active_players, dtype=np.int8),
            "hand_rank": self.get_hand_rank(self.hands[self.agent_id], self.board)
        }

    def _get_obs_player(self, player_id):
        board = self.board[:5]
        board += [None]*(5-len(board))
        return {
            "hero_hand": [self.card_as_index(c) for c in self.hands[player_id]],
            "board": [self.card_as_index(c) if c is not None else 52 for c in board],
            "stacks": np.array(self.stacks, dtype=np.float32),
            "pot": np.array([self.pot], dtype=np.float32),
            "current_bet": np.array([self.current_bet], dtype=np.float32),
            "active_players": np.array(self.active_players, dtype=np.int8),
            "hand_rank": self.get_hand_rank(self.hands[self.agent_id], self.board)
        }


    def step(self, action=None):
        """Step del agente; los oponentes reaccionan internamente"""

        print("\n=========="+ self.round_stage.upper() +"==========")
        self._log(
            f"ROUND | stage={self.round_stage} | "
            f"board={cards_int_to_str(self.board)}"
        )

        self.hand_rank = self.get_hand_rank(self.hands[self.agent_id], self.board)
        self.done = False
        ronda_done = True
        self.reward = 0

        if self.current_bet == 0 or (self.round_stage == "preflop" and self.pot == self.big_blind + self.small_blind):
            self.render()

        first = (self.dealer_pos + 2) % self.num_players
        turn_order = [(first + i) % self.num_players for i in range(self.num_players)]

        for player_id in turn_order:
            
            if not self.active_players[player_id] or self.done:
                self.all_check[player_id] = True
                continue
            
            mask = self.action_masks(player_id)
            
            # Agente
            if player_id == self.agent_id:
                if action is not None:
                    self._apply_action(self.agent_id, action)
                    print(f"Agente tomó acción: {self.ACTIONS[action]}")
                    self._log(f"AGENTE | player={player_id} | "
                                f"action={self.ACTIONS[action]} | "
                                f"stack={self.stacks[player_id]} | "
                                f"bet={self.bets[player_id]} | "
                                f"pot={self.pot}")

            # Oponentes
            elif player_id == 1:
                # opp_action = self.opponent_policies[0](self._get_obs_player(player_id))
                opp_action, _ = self.model_player1.predict(self._get_obs_player(1), action_masks=mask, deterministic=False)

                self._apply_action(player_id, opp_action)
                self._log(f"ACTION | player={player_id} (MaskPPO10) | "
                                f"action={self.ACTIONS[opp_action]} | "
                                f"stack={self.stacks[player_id]} | "
                                f"bet={self.bets[player_id]} | "
                                f"pot={self.pot}")

            elif player_id == 2:
                opp_action, _ = self.model_player2.predict(self._get_obs_player(2), action_masks=mask, deterministic=False)

                self._apply_action(player_id, opp_action)
                self._log(f"ACTION | player={player_id} (MaskPPO9) | "
                                f"action={self.ACTIONS[opp_action]} | "
                                f"stack={self.stacks[player_id]} | "
                                f"bet={self.bets[player_id]} | "
                                f"pot={self.pot}")
            elif player_id == 3:
                opp_action, _ = self.model_player3.predict(self._get_obs_player(3), action_masks=mask, deterministic=False)
                
                self._apply_action(player_id, opp_action)
                self._log(f"ACTION | player={player_id} (MaskPPO8) | "
                                f"action={self.ACTIONS[opp_action]} | "
                                f"stack={self.stacks[player_id]} | "
                                f"bet={self.bets[player_id]} | "
                                f"pot={self.pot}")
            else:
                opp_action, _ = self.model_player4.predict(self._get_obs_player(4), action_masks=mask, deterministic=False)
                
                self._apply_action(player_id, opp_action)
                self._log(f"ACTION | player={player_id} (MaskPPO3) | "
                                f"action={self.ACTIONS[opp_action]} | "
                                f"stack={self.stacks[player_id]} | "
                                f"bet={self.bets[player_id]} | "
                                f"pot={self.pot}")

            if sum(self.all_in) >= 1:
                self.done = True
                self._log(f"ALL IN | player={player_id} | "
                                f"action={self.ACTIONS[action]} | "
                                f"stack={self.stacks[player_id]} | ")
                break

            if sum(self.active_players) <= 1:
                self.done = True
                self._log(f"UN JUGADOR ACTIVO: {self.active_players}")
                break

            if self.ACTIONS[action] == "Check":
                self.all_check[player_id] = True

        self._log(f"APUESTA ACTUAL: {self.current_bet}")
        bets_activas = [self.bets[i] for i, active in enumerate(self.active_players) if active]
        ronda_done = len(set(bets_activas)) == 1 or sum(self.all_check) == 5

        for i in range(self.num_players):
            if self.stacks[i] <= 0:
                ronda_done = True
                self._log(f"UN JUGADOR CON 0 STACK: {self.stacks[i]}") 

        # Avanzar ronda
        if not self.done and ronda_done:
            self.bets = [0] * self.num_players
            self.current_bet = 0
            if self.round_stage == 'preflop':
                self.board += self.deck.draw(3)  # flop
                self.round_stage = 'flop'
                self._log(f"RONDA AVANZA A: {self.round_stage}")
            elif self.round_stage == 'flop':
                self.board += self.deck.draw(1)  # turn
                self.round_stage = 'turn'
                self._log(f"RONDA AVANZA A: {self.round_stage}")
            elif self.round_stage == 'turn':
                self.board += self.deck.draw(1)  # river
                self.round_stage = 'river'
                self._log(f"RONDA AVANZA A: {self.round_stage}")
            elif self.round_stage == 'river':
                self.done = True
                self._log(f"RONDA TERMINADA")
        
        if self.done:
            self._log(f"RESOLVER MANO")
            self._resolve_hand()
        else:
            strength = self.preflop_hand_strength(self.hands[self.agent_id])

            if not self.agent_folded:
                if self.hand_rank is not None:
                    if self.round_stage == "preflop":
                        self.reward += 5 * strength
                    else:
                        self.reward += self.HAND_RANK_REWARD[self.hand_rank] 
            else:
                if self.round_stage == "preflop":
                    if strength > 0.6:
                        self.reward -= 1.5 * strength
                        self._log("FOLD PREFLOP MANO FUERTE")
                    elif strength < 0.4:
                        self.reward += 0.4
                    else:
                        self.reward -= 0.2
                else:
                    if self.hand_rank <= 7:
                        self.reward -= 3.0 * (8 - self.hand_rank) / 2
                    elif self.hand_rank == 9:
                        self.reward += 0.5
                    elif self.hand_rank == 8 and self.round_stage == "river":
                        ranks = sorted([c % 13 for c in self.hands[self.agent_id]])
                        if max(ranks) < 10:
                            self.reward += 0.1
                    else:
                        self.reward -= 0.2

        return self._get_obs(), self.reward, self.done, False, {}
    
    def action_masks(self, player_id=None):
        if player_id is None:
            player_id = self.agent_id

        if not self.active_players[player_id]:
            return np.array([False, False, False, False, False], dtype=bool)

        stack = self.stacks[player_id]
        player_current_bet = self.bets[player_id]

        can_fold = player_current_bet < self.current_bet and stack >= 0
        can_check = self.current_bet == player_current_bet
        can_call = self.current_bet > player_current_bet
        can_bet = self.current_bet == 0 and stack >= self.big_blind

        if self.current_bet == 0:
            raise_amount = 2 * self.big_blind
        else:
            raise_amount = self.current_bet + 2 * self.big_blind

        can_raise = stack >= raise_amount

        return np.array([can_fold, can_check, can_call, can_bet, can_raise], dtype=bool)

    def _apply_action(self, player_id, action):
        if not self.active_players[player_id] or self.all_in[player_id]:
            return
        
        player_bet = self.bets[player_id]
        action_fin = ""
        stack = self.stacks[player_id]

        if action == 0 and player_bet == self.current_bet:
            action_fin = "Check"
            print(f"Jugador {player_id + 1} tomó acción: {action_fin}")
            return

        # Acción fold
        if action == 0:
            action_fin = "Fold"
            self.active_players[player_id] = False
            if player_id == self.agent_id:
                self.agent_folded = True

        # Acción check
        elif action == 1:
            action_fin = "Check"

        # Acción call
        elif action == 2:
            to_call = self.current_bet - player_bet

            if to_call <= 0:
                action_fin = "Check"
                amount = 0
            
            elif to_call >= stack:
                action_fin = "Call (All-in)"
                self.all_in[player_id] = True
                amount = stack
            else:
                action_fin = "Call"
                amount = to_call

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount

        # Acción bet
        elif action == 3:
            if self.current_bet > 0:
                return self._apply_action(player_id, 2)

            if stack <= self.big_blind:
                self.all_in[player_id] = True
                action_fin = "Bet (All-in)"
                amount = stack
            else:
                action_fin = "Bet"
                amount = self.big_blind
                

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount
            self.current_bet = self.bets[player_id]

        # Acción raise
        elif action == 4:

            if self.current_bet == 0:
                to_raise = 2 * self.big_blind
            else:
                to_raise = 2 * self.current_bet
            
            if stack + player_bet <= to_raise:
                action_fin = "Raise (All-in)"
                amount = stack
                self.all_in[player_id] = True
            else:
                action_fin = "Raise"
                amount = to_raise - player_bet

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount
            
            self.current_bet = self.bets[player_id]
        
        if player_id != 0: 
            print(f"Jugador {player_id + 1} tomó acción: {action_fin}")

    def _resolve_hand(self):
        active_hands = [(i, self.hands[i]) for i, active in enumerate(self.active_players) if active]

        self._log(f"STACKS: {self.stacks}")

        self._log(f"HERO HAND: {[Card.int_to_str(c) for c in self.hands[self.agent_id]]} - RANK: {self.evaluator.class_to_string(self.hand_rank)}")

        hands_str = {}
        for i, hand in enumerate(self.hands):
            if hand is not None:
                hands_str[f"Player {i}"] = [Card.int_to_str(c) for c in hand]
            else:
                hands_str[f"Player {i}"] = None
        
        if len(active_hands) == 1:
            winner_id, winner_hand = active_hands[0]
            self.winners = [winner_id]
            self.stacks[winner_id] += self.pot
            self._log(f"GANADOR: {self.winners[0]}")

        elif sum(self.all_in) > 0:
            self.board += self.deck.draw(5 - len(self.board))
            scores = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in active_hands]
            min_score = min([s for i,s in scores])
            self.winners = [i for i,s in scores if s==min_score]

            if self.stacks[self.agent_id] == 0:
                self.reward -= 100 * self.hand_rank / 10

            share = self.pot // len(self.winners)
            
            for w in self.winners:
                self.stacks[w] += share

            self._log(f"GANADOR(ES) TRAS ALL IN: {self.winners}")

        else:
            scores = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in active_hands]
            min_score = min([s for i,s in scores])
            self.winners = [i for i,s in scores if s==min_score]

            share = self.pot // len(self.winners)

            for w in self.winners:
                self.stacks[w] += share
            
            self._log(f"GANADOR(ES): {self.winners}")
        
        self._log(f"MANOS: {hands_str}")
        self._log(f"BOARD: {cards_int_to_str(self.board)}")
        self._log(f"POT: {self.pot}")
        self._log(f"STACKS TRAS POT: {self.stacks}")

        self.pot = 0

        stack_change = self.stacks[self.agent_id] - self.preflop_stack
        
        self.reward = stack_change / self.big_blind

        self._log(f"REWARD FINAL: {self.reward}")

    def render(self):
        print("=====================================")
        print("Board:")
        Card.print_pretty_cards(self.board)
        print("Hero Hand:")
        Card.print_pretty_cards(self.hands[self.agent_id])
        print("\nStacks:", self.stacks)
        print("Pot:", self.pot)
        print("Current Bet:", self.current_bet)
        print("Active Players:", self.active_players)
        print("\n")