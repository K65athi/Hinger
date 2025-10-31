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

CELL_SIZE = 60
MARGIN = 5
FPS = 30
BOARD_SIZE = (4, 5)
AI_MODE = "alphabeta"

BG_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
ACTIVE_CELL_COLOR = (100, 200, 100)
EMPTY_CELL_COLOR = (200, 100, 100)
TEXT_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 255, 0)

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
    

"""
------------------------------------
Tester function for the State class
------------------------------------
"""

def tester():
    screen = pygame.display.set_mode((BOARD_SIZE[1] * CELL_SIZE, BOARD_SIZE[0] * CELL_SIZE + 50))
    pygame.display.set_caption("Hinger State Tester")

    # Example test case for the State class
    test_grid = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]

    state = State(test_grid)
    agent = Agent(size=BOARD_SIZE, name="TesterAgent")
    human_turn = True
    selected_cell = None
    running = True

    while running:

        screen.fill(BG_COLOR)
        draw_board(screen, state, selected_cell)

        msg = "Human's Turn" if human_turn else "AI's Turn"
        text = font.render(msg, True, TEXT_COLOR)
        screen.blit(text, (10, BOARD_SIZE[0] * CELL_SIZE + 10))
        pygame.display.flip()

        # Human
        if human_turn:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row, col = y // CELL_SIZE, x // CELL_SIZE
                    if (row < len(state.grid)) and col < (len(state.grid[0])):    
                        if state.grid[row][col] > 0:
                            
                            # Check if a hinger move was played
                            regionsBefore = state.numRegions()
                            state.grid[row][col] -= 1
                            regionsAfter = state.numRegions()
                            if regionsAfter > regionsBefore:
                                winner = "Human"
                                running = False
                                break

                            # Pass turn
                            human_turn = False

        
        # AI
        else:
            pygame.display.flip()
            regionsBefore = state.numRegions()
            move = agent.move(state, mode = AI_MODE)
            if move:
                r, c = move
                state.grid[r][c] -= 1
                regionsAfter = state.numRegions()
                if regionsAfter > regionsBefore:
                    winner = "AI"
                    running = False
                    break
            human_turn = True

        # Check for draw
        if all(cell == 0 for row in state.grid for cell in row):
            winner = None
            running = False
        
        clock.tick(FPS)

    # Display result
    screen.fill(BG_COLOR)
    if winner == "Human":
        result = "Human Wins! (Hinger Found)"
    elif winner == "AI":
        result = "AI Wins! (Hinger Found)"
    else:
        result = "It's a Draw! (No Moves Left)"
    text = font.render(result, True, TEXT_COLOR)
    screen.blit(text, (BOARD_SIZE[1] * CELL_SIZE // 2 - 140, BOARD_SIZE[0] * CELL_SIZE // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()       
    

if __name__ == "__main__":
    tester()