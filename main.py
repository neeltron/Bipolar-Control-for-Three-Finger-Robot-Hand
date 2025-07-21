# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:37:20 2025

@author: neeltron
"""

import pygame
import serial
import sys
import numpy as np
from collections import deque

# === Setup ===
pygame.init()
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Joystick Positioning')
font = pygame.font.SysFont(None, 24)

port = serial.Serial('COM7', 9600)

# === Joystick buffer and helpers ===
BUFFER_SIZE = 10
buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]
estimate_table = [{} for _ in range(3)]

def read_joystick_data(port):
    try:
        line = port.readline().decode('utf-8').strip()
        values = line.split(',')
        if len(values) == 9:
            x1, y1, b1 = int(values[0]), int(values[1]), int(values[2])
            x2, y2, b2 = int(values[3]), int(values[4]), int(values[5])
            x3, y3, b3 = int(values[6]), int(values[7]), int(values[8])
            return (x1, y1, b1), (x2, y2, b2), (x3, y3, b3)
    except Exception as e:
        print(f'Error reading data: {e}')
    return None, None, None

def normalize(x, y):
    return (x - 512) / 512, (y - 512) / 512

def compute_vector_field(positions):
    if len(positions) < 2:
        return None, None
    prev = np.array(positions[-2])
    curr = np.array(positions[-1])
    delta = curr - prev
    return prev, delta

def estimate_vector(x, joy_index):
    norm_x = tuple(np.clip(x, -1, 1))
    return np.array(estimate_table[joy_index].get(norm_x, [0, 0]))

def compute_jacobian(x, joy_index):
    epsilon = 0.01
    J = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = epsilon
        f_plus = estimate_vector(x + dx, joy_index)
        f_minus = estimate_vector(x - dx, joy_index)
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    return J

# === Main Loop ===
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    js1, js2, js3 = read_joystick_data(port)
    if js1 and js2 and js3:
        screen.fill((30, 30, 30))
        pygame.draw.line(screen, (100, 100, 100), (0, height // 2), (width, height // 2))

        js1_flipped = (1023 - js1[0], 1023 - js1[1], js1[2])
        js2_flipped = (1023 - js2[0], 1023 - js2[1], js2[2])

        for i, (x, y, button) in enumerate([js3, js2_flipped, js1_flipped]):
            xpos = int((x / 1023) * (width / 3 - 50) + 25 + i * width / 3)
            ypos = int((y / 1023) * (height - 50) + 25)
            color = (255, 0, 0) if button == 0 else (0, 255, 0)
            pygame.draw.circle(screen, color, (xpos, ypos), 15)
            pygame.draw.line(screen, (100, 100, 100), (int(width / 3 * (i + 0.5)), 0), (int(width / 3 * (i + 0.5)), height))

            norm_x, norm_y = normalize(x, y)
            buffers[i].append((norm_x, norm_y))
            pos, delta = compute_vector_field(buffers[i])
            if pos is not None:
                estimate_table[i][tuple(pos)] = delta
                J = compute_jacobian(np.array(pos), i)
                eigvals = np.linalg.eigvals(J)
                print(f"Joystick {i} Jacobian:\n{J}")
                print(f"Eigenvalues: {eigvals}\n")

        pygame.display.flip()
