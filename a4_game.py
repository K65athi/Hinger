#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

GUI Utilising Pygame for Task 4. Implements a working game between two agents or humans as well as a timer function,
and turn labels.

Game options can be adjusted in the GAME CONFIGURATION section.

@author: B6 (100393447, 100361094, and 100387788)
@date:   21/10/2025

"""

import pygame
import sys
from a1_state import State
from a3_agent import Agent, _diff_move  # add _diff_move here


"""
GAME CONFIGURATION HERE
"""
BOARD_SIZE = (4, 5) # Doesn't seem like this is changeable.
GAME_MODE = 0 # 0: AI vs AI, 1: Human vs AI, 2: AI vs Human, 3: Human vs Human
AI_1 = Agent(size=BOARD_SIZE, name="AI-Agent Yu Eeyai")
AI_2 = Agent(size=BOARD_SIZE, name="AI-Agent Core Suwerk")
AI_DELAY = 1000  # milliseconds delay for AI moves
TURN_TIME_LIMIT = 10  # seconds per turn
AI_MODE = "alphabeta" # Unsure if this affects anything

match (GAME_MODE):
    case 0:
        agentA = AI_1
        agentB = AI_2
    case 1:
        agentA = None
        agentB = AI_2
    case 2:
        agentA = AI_1
        agentB = None
    case 3:
        agentA = None
        agentB = None

"""
Pygame settings and values
"""

CELL_SIZE = 60
MARGIN = 5
FPS = 30
TOP_OFFSET = 50
BOTTOM_BAR_HEIGHT = 50

WINDOW_WIDTH = BOARD_SIZE[1] * CELL_SIZE
WINDOW_HEIGHT = BOARD_SIZE[0] * CELL_SIZE + TOP_OFFSET + BOTTOM_BAR_HEIGHT

BG_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
ACTIVE_CELL_COLOR = (100, 200, 100)
EMPTY_CELL_COLOR = (200, 100, 100)
TEXT_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 255, 0)

"""
Pygame Initialization
"""

pygame.init()
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()

def draw_board(screen, state, selected_cell=None, offset_y=0):
    rows, cols = len(state.grid), len(state.grid[0])
    for i in range(rows):
        for j in range(cols):
            x, y = j * CELL_SIZE, i * CELL_SIZE + offset_y
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)

            if state.grid[i][j] > 0:
                color = ACTIVE_CELL_COLOR
            else:
                color = EMPTY_CELL_COLOR
            pygame.draw.rect(screen, color, rect)

            # On highlight
            if selected_cell == (i, j):
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)

            if state.grid[i][j] > 0:
                text = font.render(str(state.grid[i][j]), True, TEXT_COLOR)
                screen.blit(text, (x + CELL_SIZE // 2 - text.get_width() // 2, y + CELL_SIZE // 2 - text.get_height() // 2))
    
def play(state, agentA = None, agentB = None):
    """"
    Plays a game between two agents or human. (None is Human)
    """

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hinger Game")

    humanA = agentA is None
    humanB = agentB is None
    current_agent = agentA
    current_name = "Player A" if humanA else getattr(agentA, 'name', 'Agent A')
    other_agent = agentB
    other_name = "Player B" if humanB else getattr(agentB, 'name', 'Agent B')

    running = True
    winner = None
    selected_cell = None
    turn_start_time = pygame.time.get_ticks()

    while running:
        screen.fill(BG_COLOR)

        # Timer at the top
        elapsed_time = pygame.time.get_ticks() - turn_start_time
        remaining_time = max(0, TURN_TIME_LIMIT - elapsed_time // 1000)
        timer_msg = font.render(f"Time Left: {remaining_time}s", True, TEXT_COLOR)
        screen.blit(timer_msg, (10, 10))

        # Draw the board
        draw_board(screen, state, selected_cell, offset_y=TOP_OFFSET)

        # Display turn info
        pygame.draw.rect(screen, GRID_COLOR, (0, TOP_OFFSET + BOARD_SIZE[0] * CELL_SIZE, BOARD_SIZE[1] * CELL_SIZE, BOTTOM_BAR_HEIGHT))

        # Player info
        text_turn = font.render(f"{current_name}'s Turn", True, TEXT_COLOR)
        screen.blit(text_turn, (10, TOP_OFFSET + BOARD_SIZE[0] * CELL_SIZE + BOTTOM_BAR_HEIGHT //4))

        

        pygame.display.flip()

        # Check for time out
        no_moves = True
        for _ in state.moves():
            no_moves = False
            break
        if no_moves:
            # Current player has no move; previous player wins
            winner = other_name
            running = False
            break

        # Check for time out
        if remaining_time <= 0:
            winner = other_name
            running = False
            break

        move = None

        # Human
        if (humanA and current_agent == agentA) or (humanB and current_agent == agentB):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row, col = (y - TOP_OFFSET) // CELL_SIZE, x // CELL_SIZE
                    if (0 <= row < len(state.grid)) and (0 <= col < len(state.grid[0])):    
                        if state.grid[row][col] <= 0:
                            # Invalid move, therefore opponent should win
                            winner = other_name
                            running = False
                            break
                        move = (row, col)
            if move is None:
                clock.tick(FPS)
                continue
        
        # AI
        else:
            pygame.time.delay(AI_DELAY)
            import time
            t0 = time.perf_counter()
            move = current_agent.move(state, mode=AI_MODE)
            dt = time.perf_counter() - t0
            print(f"[AI] {current_name} used {AI_MODE} in {dt:.4f}s -> {move}")
    
            if move is None or state.grid[move[0]][move[1]] <= 0:
                winner = other_name
                running = False
                break


        # Make the move
        regionsBefore = state.numRegions()
        cell_before = state.grid[move[0]][move[1]]

        # Find the matching child produced by the rules engine
        next_state = None
        for child in state.moves():
            try:
                if _diff_move(state, child) == move:
                    next_state = child
                    break
            except Exception:
                continue

        if next_state is None:
            # If we couldn't map, the move is invalid in terms of game rules => other side wins
            winner = other_name
            running = False
            break

        regionsAfter = next_state.numRegions()

        # Hinger win: click on a 1 that increases region count
        if cell_before == 1 and regionsAfter > regionsBefore:
            winner = current_name
            print(f"💥 Hinger move by {current_name}! They win!")
            running = False
            state = next_state  # show the winning state
            break
        else:
            state = next_state  # normal update

            # Swap turn
            current_agent, other_agent = other_agent, current_agent
            current_name,  other_name  = other_name,  current_name
            turn_start_time = pygame.time.get_ticks()
            clock.tick(FPS)
            continue

    
    # Display result
    screen.fill(BG_COLOR)
    if winner is None:
        result = "It's a Draw! (No Moves Left)"
    else:
        result = f"{winner} Wins!"
    text = font.render(result, True, TEXT_COLOR)
    screen.blit(text, (BOARD_SIZE[1] * CELL_SIZE // 2 - 100, BOARD_SIZE[0] * CELL_SIZE // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()
    return winner

"""
------------------------------------
Tester function
------------------------------------
"""

def tester():

    test_grid = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]

    state = State(test_grid)
    play(state, agentA, agentB)  # AI vs Human

if __name__ == "__main__":
    tester()