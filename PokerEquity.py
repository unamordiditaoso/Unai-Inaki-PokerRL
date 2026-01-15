import random
import time
from treys import Card, Evaluator, Deck

# pip install treys

def parse_cards(str_cards):
    """
    Convierte una lista de strings tipo 'As','Kd','Th','2c' a la representación de treys.
    Notación aceptada:
       - Rango de rangos: A,K,Q,J,T,9,...2
       - Sufijo de palo: s,h,d,c (spades, hearts, diamonds, clubs)
    Ejemplo: ["As", "Kd", "Th"]
    """
    treys_cards = []
    for s in str_cards:
        s = s.strip()
        # treys acepta e.g. "As", "Td", "9c"
        # Card.new también acepta "Ah" etc.
        try:
            treys_cards.append(Card.new(s))
        except Exception as ex:
            raise ValueError(f"Formato de carta inválido: '{s}'. Ejemplo válido: 'As', 'Td', '7c'") from ex
    return treys_cards

def estimate_equity(hero_str, board_str=None, num_opponents=4, iters=20000, seed=None):
    """
    Estima win/tie probabilities para la mano 'hero_str' frente a num_opponents rivales.
    - hero_str: list de 2 strings, e.g. ["As","Kd"]
    - board_str: list de 0..5 strings (cartas comunes), e.g. ["2h","7d","Jc"]
    - num_opponents: número de oponentes (excluye al hero). Para 5 jugadores en mesa => num_opponents = 4
    - iters: número de simulaciones Monte Carlo
    - seed: semilla aleatoria opcional (int)

    Retorna: (win_prob, tie_prob, loss_prob)
    """
    if board_str is None:
        board_str = []

    if len(hero_str) != 2:
        raise ValueError("hero_str debe contener exactamente 2 cartas.")

    if not (0 <= len(board_str) <= 5):
        raise ValueError("board_str debe contener entre 0 y 5 cartas.")

    if seed is not None:
        random.seed(seed)

    evaluator = Evaluator()

    # parseo
    hero = parse_cards(hero_str)
    board = parse_cards(board_str)

    wins = 0
    ties = 0
    losses = 0

    start = time.time()

    for i in range(iters):
        deck = Deck() 
        used = set(hero + board)
        full_deck = [c for c in deck.cards if c not in used]
        # barajar
        random.shuffle(full_deck)

        opps = []
        idx = 0
        for _ in range(num_opponents):
            opp_hand = [full_deck[idx], full_deck[idx + 1]]
            idx += 2
            opps.append(opp_hand)

        cards_needed = 5 - len(board)
        board_complete = list(board)  # copia
        board_complete += full_deck[idx: idx + cards_needed]

        hero_score = evaluator.evaluate(hero, board_complete)
        opp_scores = [evaluator.evaluate(opp, board_complete) for opp in opps]

        best_score = min([hero_score] + opp_scores)

        winners = 1 if hero_score == best_score else 0
        winners += sum(1 for s in opp_scores if s == best_score)

        if hero_score == best_score:
            if winners == 1:
                wins += 1
            else:
                ties += 1
        else:
            losses += 1

    elapsed = time.time() - start

    win_prob = wins / iters
    tie_prob = ties / iters
    loss_prob = losses / iters

    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_prob": win_prob,
        "tie_prob": tie_prob,
        "loss_prob": loss_prob,
        "iterations": iters,
        "time_s": elapsed,
        "per_s": iters / elapsed if elapsed > 0 else None
    }

def pretty_card_list(str_cards):
    return " ".join(str_cards)

if __name__ == "__main__":
    hero = ["As", "Ks"]
    board = []
    opponents = 4

    print("Calculando equity Monte Carlo ...")
    print(f"Hero: {pretty_card_list(hero)}")
    print(f"Board (parcial): {pretty_card_list(board)}")
    print(f"Oponentes: {opponents}")
    res = estimate_equity(hero, board, num_opponents=opponents, iters=50000, seed=42)

    print("\nResultados (estimación):")
    print(f"Iteraciones: {res['iterations']}")
    print(f"Tiempo: {res['time_s']:.2f} s ({res['per_s']:.0f} simul/s)")
    print(f"Gana: {res['wins']}  ({res['win_prob']*100:.2f} %)")
    print(f"Empata: {res['ties']}  ({res['tie_prob']*100:.2f} %)")
    print(f"Pierde: {res['losses']}  ({res['loss_prob']*100:.2f} %)")