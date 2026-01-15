import pygame
from PokerEnv import Poker5EnvFull
from treys import Card, Deck, Evaluator
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# --- Inicializar entorno ---
model1 = PPO.load("checkpoints_poker/PPO7/ppo_poker_final.zip")
model2 = MaskablePPO.load("checkpoints_poker/MaskPPO2/ppo_poker_final.zip")
model3 = MaskablePPO.load("checkpoints_poker/MaskPPO1/ppo_poker_final.zip")
model4 = PPO.load("checkpoints_poker/PPO6/ppo_poker_final.zip")

env = Poker5EnvFull()
obs = env.__init__(model_player1=model1, model_player2=model2, model_player3=model3, model_player4=model4)
obs = env.partial_reset()
model = MaskablePPO.load("checkpoints_poker/MaskPPO2/ppo_poker_final.zip")

# --- Pygame ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("PokerRL Simulator")
clock = pygame.time.Clock()

# --- Constantes ---
CARD_SIZE = (80, 120)
RANKS = "23456789TJQKA"
SUITS = "cdhs"

# --- Posiciones jugadores ---
PLAYER_POSITIONS = [
    (175, 450),  # Hero
    (75, 100),  # Jugador 1
    (325, 50),   # Jugador 2
    (575, 100),  # Jugador 3
    (475, 450),  # Jugador 4
]

# --- Cartas comunitarias ---
BOARD_POSITION = (175, 250)
BOARD_SPACING = 100

# --- Cargar imágenes ---
card_images = {}
BACK_CARD = pygame.image.load("cartas/backBlack.png").convert_alpha()
BACK_CARD = pygame.transform.scale(BACK_CARD, CARD_SIZE)

chip_img = pygame.image.load("cartas/chip.png").convert_alpha()
chip_img = pygame.transform.scale(chip_img, (18, 18))


def load_card(card_str):
    if card_str not in card_images:
        img = pygame.image.load(f"cartas/{card_str}.png").convert_alpha()
        card_images[card_str] = pygame.transform.scale(img, CARD_SIZE)
    return card_images[card_str]



def draw_stacks(obs):
    font = pygame.font.SysFont(None, 24)
    
    stacks = obs["stacks"]

    for i, (x, y) in enumerate(PLAYER_POSITIONS):
        stack = int(stacks[i])

        # Icono ficha
        screen.blit(chip_img, (x + CARD_SIZE[0] + 38, y - 30))

        # Texto stack
        text = font.render(str(stack), True, (255, 255, 255))
        screen.blit(text, (x + CARD_SIZE[0] + 60, y - 28))

def draw_pot(obs):
    font = pygame.font.SysFont(None, 24)

    pot = int(obs["pot"][0])
    x, y = 400, 300

    screen.blit(chip_img, (x - 26, y - 80))
    text = font.render(f"Pot: {pot}", True, (255, 215, 0))
    screen.blit(text, (x, y - 78))

def showdown():

    screen.fill((0, 128, 0))

    activePlayers = obs.get("active_players", [True]*5)
    
    font = pygame.font.SysFont(None, 28)
    
    for i, pos in enumerate(PLAYER_POSITIONS):

        x, y = pos
        if i == 0:
            label = font.render(f"Agente", True, (255, 255, 0))
        else: 
            label = font.render(f"Jugador {i + 1}", True, (255, 255, 0))
        label_rect = label.get_rect(center=(x + CARD_SIZE[0] - 20, y - 20))
        screen.blit(label, label_rect)

        if not activePlayers[i]:
            continue 

        hand = env.hands[i]        
        for j, idx in enumerate(hand):
            card_str = Card.int_to_str(idx)
            screen.blit(load_card(card_str), (x + j*100, y))

    board_idx = env.board

    for j, idx in enumerate(board_idx):
        x = BOARD_POSITION[0] + j*BOARD_SPACING
        y = BOARD_POSITION[1]

        if idx is not None:
            card_str = Card.int_to_str(idx)
            screen.blit(load_card(card_str), (x, y))
        else:
            screen.blit(BACK_CARD, (x, y))

    font = pygame.font.SysFont(None, 40)
    winner_str = " y ".join("Agente" if w == 0 else "Jugador " + str(w+1) for w in env.winners)
    if len(env.winners) > 1:
        text = f"Ganan {winner_str}"
    else:
        text = f"Gana {winner_str}"
    label = font.render(text, True, (255, 215, 0))
    screen.blit(label, label.get_rect(center=(400, 400)))

    draw_stacks(obs)
    draw_pot(obs)

    pygame.display.flip()
    pygame.time.delay(3000)

ACTION_DELAY = 500
last_action_time = 0
action = None

# --- Loop principal ---
running = True
while running:
    clock.tick(30)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    now = pygame.time.get_ticks()

    if now - last_action_time > ACTION_DELAY:
        mask = env.action_masks(0)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        last_action_time = now

        done = terminated or truncated
            
        if done:
            showdown()
            
            obs = env.partial_reset()



    screen.fill((0, 128, 0))

    activePlayers = obs.get("active_players", [True]*5)

    for i, pos in enumerate(PLAYER_POSITIONS):
        
        x, y = pos

        font = pygame.font.SysFont(None, 28)

        if i == 0:
            label = font.render(f"Agente", True, (255, 255, 0))
        else:
            label = font.render(f"Jugador {i + 1}", True, (255, 255, 0))
        
        label_rect = label.get_rect(center=(x + CARD_SIZE[0] - 20, y - 20))
        screen.blit(label, label_rect)
        
        if not activePlayers[i]:
            continue

        if i == 0:
            hero_hand_idx = env.hands[i]

            for j, idx in enumerate(hero_hand_idx):
                card_str = Card.int_to_str(idx)
                screen.blit(load_card(card_str), (x + j*100, y))
        else: 
            for j in range(2):
                screen.blit(BACK_CARD, (x + j*100, y))       

    board_idx = env.board
    
    for j, idx in enumerate(board_idx):
        x = BOARD_POSITION[0] + j*BOARD_SPACING
        y = BOARD_POSITION[1]

        if idx is not None:
            card_str = Card.int_to_str(idx)
            screen.blit(load_card(card_str), (x, y))

    draw_stacks(obs)
    draw_pot(obs)

    pygame.time.delay(200)

    pygame.display.flip()

pygame.quit()
