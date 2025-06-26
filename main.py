# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:37:20 2025

@author: neeltron
"""

import pygame
import serial
import sys

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

pygame.init()
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Joystick Positioning')

# Serial port configuration
port = serial.Serial('COM7', 9600)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    js1, js2, js3 = read_joystick_data(port)
    if js1 and js2 and js3:
        screen.fill((30, 30, 30))
        # Draw horizontal line
        pygame.draw.line(screen, (100, 100, 100), (0, height // 2), (width, height // 2))
        for i, (x, y, button) in enumerate([js1, js2, js3]):
            xpos = int((x / 1023) * (width / 3 - 50) + 25 + i * width / 3)
            ypos = int((y / 1023) * (height - 50) + 25)
            color = (255, 0, 0) if button == 0 else (0, 255, 0)
            pygame.draw.circle(screen, color, (xpos, ypos), 15)
            pygame.draw.line(screen, (100, 100, 100), (int(width / 3 * (i + 0.5)), 0), (int(width / 3 * (i + 0.5)), height))
        pygame.display.flip()
