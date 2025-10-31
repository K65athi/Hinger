#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B6 (100393447, 100361094, and 100387788)
@date:   10/10/2025

"""

import pygame
import sys
from a1_state import State
from a3_agent import Agent

"""
GAME CONFIGURATION HERE
"""
BOARD_SIZE = (4, 5)
GAME_MODE = 1 # 0: AI vs AI, 1: Human vs AI, 2: AI vs Human, 3: Human vs Human
AI_1 = Agent(size=BOARD_SIZE, name="AI-Agent Yu Eeyai")
AI_2 = Agent(size=BOARD_SIZE, name="AI-Agent Core Suwerk")

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
AI_MODE = "alphabeta"

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

def draw_board(screen, state, selected_cell=None):
    rows, cols = len(state.grid), len(state.grid[0])
    for i in range(rows):
        for j in range(cols):
            x, y = j * CELL_SIZE, i * CELL_SIZE
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

    screen = pygame.display.set_mode((BOARD_SIZE[1] * CELL_SIZE, BOARD_SIZE[0] * CELL_SIZE + 50))
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

    while running:
        screen.fill(BG_COLOR)
        draw_board(screen, state, selected_cell)
        msg = f"{current_name}'s Turn"
        text = font.render(msg, True, TEXT_COLOR)
        screen.blit(text, (10, BOARD_SIZE[0] * CELL_SIZE + 10))
        pygame.display.flip()

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
                    row, col = y // CELL_SIZE, x // CELL_SIZE
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
            pygame.time.delay(500)  # Let's you see the board before AI moves
            move = current_agent.move(state, mode = AI_MODE)
            if move is None or state.grid[move[0]][move[1]] <= 0:
                # Invalid move, therefore opponent should win
                winner = other_name
                running = False
                break

        # Make the move
        regionsBefore = state.numRegions()
        state.grid[move[0]][move[1]] -= 1
        regionsAfter = state.numRegions()

        if regionsAfter > regionsBefore:
            winner = current_name
            running = False
            
        # Check draw
        elif all(cell == 0 for row in state.grid for cell in row):
            winner = None
            running = False

        # Switch turns
        current_agent, other_agent = other_agent, current_agent
        current_name, other_name = other_name, current_name
        clock.tick(FPS)
    
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