"""
Professional Visualizer for NCP Drive-Modulated Soliter Training.

Features:
- Transparent agent body (can see through)
- Sense circles showing ray distances
- Resource detection/consumption zones
- Resource availability indicators
- Season effects and availability
- Complete dashboard with drives
- Day/night cycle visualization
"""

import pygame
import numpy as np
from typing import Dict, Optional

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 50, 50)
GREEN = (100, 255, 100)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PURPLE = (200, 100, 255)


class ProfessionalVisualizer:
    """
    Professional visualizer with transparent agent and complete information display.
    """
    
    def __init__(self, world_width: int, world_height: int, scale: int = 4):
        pygame.init()
        self.scale = scale
        self.width = world_width * scale
        self.height = world_height * scale + 200  # Extra space for dashboard
        self.world_height = world_height * scale
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NCP Drive-Modulated Agent - Professional View")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
    
    def draw(self, agent, resources, world, trainer, cycle, step, total_consumptions, last_reward_details):
        """Draw complete visualization."""
        is_night = world.is_night()
        season = world.get_season()
        
        # Background based on time/season
        self._draw_background(is_night, season)
        
        # Resources with zones
        self._draw_resources(resources, season, is_night)
        
        # Agent with transparency and sense circles
        self._draw_agent(agent, is_night)
        
        # Dashboard
        self._draw_dashboard(
            agent, trainer, world, season, is_night,
            cycle, step, total_consumptions, last_reward_details
        )
        
        pygame.display.flip()
    
    def _draw_background(self, is_night: bool, season):
        """Draw background with day/night and season effects."""
        if is_night:
            # Night backgrounds
            if season.name == 'SUMMER':
                bg = (20, 20, 60)  # Dark blue
            elif season.name == 'WINTER':
                bg = (10, 10, 10)  # Black
            elif season.name == 'SPRING':
                bg = (30, 30, 50)  # Dark grey-blue
            else:  # AUTUMN
                bg = (40, 30, 30)  # Dark brown
        else:
            # Day backgrounds
            if season.name == 'SUMMER':
                bg = (255, 255, 220)  # Warm white
            elif season.name == 'WINTER':
                bg = (240, 245, 255)  # Cool white
            elif season.name == 'SPRING':
                bg = (245, 255, 245)  # Green-tinted white
            else:  # AUTUMN
                bg = (255, 250, 240)  # Warm white
        
        self.screen.fill(bg)
    
    def _draw_resources(self, resources: Dict, season, is_night: bool):
        """Draw resources with detection/consumption zones and availability."""
        
        # FOOD (Feeders) - Green
        for feeder in resources.get('feeders', []):
            pos = (int(feeder.position[0] * self.scale), 
                   int(feeder.position[1] * self.scale))
            
            # Get availability
            depletion = feeder.get_depletion_ratio()
            available = depletion > 0.1
            
            # Detection zone (faint)
            detection_r = int(feeder.get_detection_radius(False) * self.scale)
            color = (100, 200, 100) if available else (50, 100, 50)
            pygame.draw.circle(self.screen, color, pos, detection_r, 1)
            
            # Consumption zone (visible)
            consumption_r = int(feeder.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, color, pos, consumption_r, 2)
            
            # Center dot
            center_color = GREEN if available else DARK_GRAY
            pygame.draw.circle(self.screen, center_color, pos, 6, 0)
            
            # Availability %
            if depletion < 1.0:
                percent = int(depletion * 100)
                text_color = BLACK if not is_night else WHITE
                text = self.tiny_font.render(f'{percent}%', True, text_color)
                self.screen.blit(text, (pos[0] + 10, pos[1] - 10))
        
        # WATER (Fountains) - Blue
        for fountain in resources.get('fountains', []):
            pos = (int(fountain.position[0] * self.scale),
                   int(fountain.position[1] * self.scale))
            
            depletion = fountain.get_depletion_ratio()
            available = depletion > 0.1
            
            # Detection zone
            detection_r = int(fountain.get_detection_radius(False) * self.scale)
            color = (100, 150, 255) if available else (50, 75, 127)
            pygame.draw.circle(self.screen, color, pos, detection_r, 1)
            
            # Consumption zone
            consumption_r = int(fountain.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, color, pos, consumption_r, 2)
            
            # Center dot
            center_color = BLUE if available else DARK_GRAY
            pygame.draw.circle(self.screen, center_color, pos, 6, 0)
            
            # Availability %
            if depletion < 1.0:
                percent = int(depletion * 100)
                text_color = BLACK if not is_night else WHITE
                text = self.tiny_font.render(f'{percent}%', True, text_color)
                self.screen.blit(text, (pos[0] + 10, pos[1] - 10))
        
        # HEAT (Heaters) - Orange (check seasonal availability!)
        for heater in resources.get('heaters', []):
            pos = (int(heater.position[0] * self.scale),
                   int(heater.position[1] * self.scale))
            
            # Check if heat is available this season
            heat_active = season.name in ['WINTER', 'AUTUMN', 'SPRING']
            depletion = heater.get_depletion_ratio() if heat_active else 0.0
            available = depletion > 0.1 and heat_active
            
            # Detection zone
            detection_r = int(heater.get_detection_radius(False) * self.scale)
            if heat_active:
                color = (255, 150, 50) if available else (127, 75, 25)
            else:
                color = (100, 100, 100)  # Gray when inactive
            pygame.draw.circle(self.screen, color, pos, detection_r, 1)
            
            # Consumption zone
            consumption_r = int(heater.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, color, pos, consumption_r, 2)
            
            # Center X
            x_size = 8
            center_color = ORANGE if available else DARK_GRAY
            pygame.draw.line(self.screen, center_color,
                           (pos[0]-x_size, pos[1]-x_size),
                           (pos[0]+x_size, pos[1]+x_size), 2)
            pygame.draw.line(self.screen, center_color,
                           (pos[0]+x_size, pos[1]-x_size),
                           (pos[0]-x_size, pos[1]+x_size), 2)
            
            # Show "INACTIVE" in summer
            if not heat_active:
                text_color = BLACK if not is_night else WHITE
                text = self.tiny_font.render('OFF', True, text_color)
                self.screen.blit(text, (pos[0] + 10, pos[1] - 10))
            elif depletion < 1.0:
                percent = int(depletion * 100)
                text_color = BLACK if not is_night else WHITE
                text = self.tiny_font.render(f'{percent}%', True, text_color)
                self.screen.blit(text, (pos[0] + 10, pos[1] - 10))
    
    def _draw_agent(self, agent, is_night: bool):
        """
        Draw agent as a small fly-like creature.
        - Tiny body (3-4 pixels)
        - Semi-transparent wings
        - Heading indicator
        """
        agent_pos = (int(agent.position[0] * self.scale),
                     int(agent.position[1] * self.scale))
        
        # Fly body is TINY (agent.radius=10 → 40 pixels, but we draw 3-4 pixel body)
        body_size = 3 if self.scale <= 4 else 4
        
        # Colors
        if not agent.is_alive:
            body_color = (80, 80, 80)
            wing_color = (100, 100, 100, 60)  # Gray, very transparent
            heading_color = (60, 60, 60)
        elif is_night:
            body_color = (200, 200, 255)  # Light blue (visible at night)
            wing_color = (150, 150, 200, 80)  # Blue-ish transparent
            heading_color = YELLOW
        else:
            body_color = (40, 40, 40)  # Dark body (visible during day)
            wing_color = (200, 200, 200, 60)  # Light gray transparent
            heading_color = RED
        
        # Create surface for transparent wings
        wing_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
        wing_center = (40, 40)
        
        # Wing shape: two ovals at angles from body
        wing_length = 12
        wing_width = 6
        
        # Left wing
        left_wing_angle = agent.heading + np.pi/3  # 60° left
        left_end = (
            int(wing_center[0] + wing_length * np.cos(left_wing_angle)),
            int(wing_center[1] + wing_length * np.sin(left_wing_angle))
        )
        pygame.draw.ellipse(
            wing_surface, wing_color,
            (left_end[0]-wing_width, left_end[1]-wing_length//2, wing_width*2, wing_length),
            0
        )
        
        # Right wing  
        right_wing_angle = agent.heading - np.pi/3  # 60° right
        right_end = (
            int(wing_center[0] + wing_length * np.cos(right_wing_angle)),
            int(wing_center[1] + wing_length * np.sin(right_wing_angle))
        )
        pygame.draw.ellipse(
            wing_surface, wing_color,
            (right_end[0]-wing_width, right_end[1]-wing_length//2, wing_width*2, wing_length),
            0
        )
        
        # Blit wings to screen
        wing_rect = wing_surface.get_rect(center=agent_pos)
        self.screen.blit(wing_surface, wing_rect)
        
        # Draw tiny body (filled circle)
        pygame.draw.circle(self.screen, body_color, agent_pos, body_size, 0)
        
        # Heading indicator (short line from body)
        heading_len = 15
        heading_end = (
            agent_pos[0] + int(heading_len * np.cos(agent.heading)),
            agent_pos[1] + int(heading_len * np.sin(agent.heading))
        )
        pygame.draw.line(self.screen, heading_color, agent_pos, heading_end, 2)
        
        # Tiny dot at arrow tip
        pygame.draw.circle(self.screen, heading_color, heading_end, 2, 0)
    
    def _draw_dashboard(self, agent, trainer, world, season, is_night: bool,
                       cycle: int, step: int, total_consumptions: int,
                       last_reward_details: Dict):
        """Draw comprehensive dashboard."""
        y = self.world_height + 10
        text_color = BLACK if not is_night else WHITE
        
        # Row 1: Cycle, Step, Consumptions
        text = self.font.render(
            f"Cycle: {cycle}  Step: {step}  Consumptions: {total_consumptions}",
            True, text_color
        )
        self.screen.blit(text, (10, y))
        y += 25
        
        # Row 2: Season (colored) + Day/Night
        season_colors = {
            'SUMMER': ORANGE,
            'WINTER': BLUE,
            'SPRING': GREEN,
            'AUTUMN': (180, 100, 0)
        }
        season_color = season_colors.get(season.name, text_color)
        
        text = self.font.render(f"Season: {season.name}", True, season_color)
        self.screen.blit(text, (10, y))
        
        # Day/Night indicator
        time_text = "NIGHT" if is_night else "DAY"
        text = self.font.render(time_text, True, text_color)
        self.screen.blit(text, (250, y))
        y += 30
        
        # Row 3: Vitals (bars)
        bar_width = 150
        bar_height = 12
        
        # Energy
        energy_pct = min(1.0, agent.energy / 100.0)
        pygame.draw.rect(self.screen, DARK_GRAY, (10, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, GREEN, (10, y, int(bar_width * energy_pct), bar_height))
        text = self.small_font.render(f"Energy: {agent.energy:.0f}", True, text_color)
        self.screen.blit(text, (170, y))
        y += 18
        
        # Hydration
        hydration_pct = min(1.0, agent.hydration / 100.0)
        pygame.draw.rect(self.screen, DARK_GRAY, (10, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, BLUE, (10, y, int(bar_width * hydration_pct), bar_height))
        text = self.small_font.render(f"Hydration: {agent.hydration:.0f}", True, text_color)
        self.screen.blit(text, (170, y))
        y += 18
        
        # Temperature
        temp_pct = min(1.0, max(0.0, agent.temperature / 100.0))
        pygame.draw.rect(self.screen, DARK_GRAY, (10, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, RED, (10, y, int(bar_width * temp_pct), bar_height))
        text = self.small_font.render(f"Temp: {agent.temperature:.1f}°C", True, text_color)
        self.screen.blit(text, (170, y))
        y += 18
        
        # Wakefulness
        wake_pct = min(1.0, agent.wakefulness)
        pygame.draw.rect(self.screen, DARK_GRAY, (10, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, CYAN, (10, y, int(bar_width * wake_pct), bar_height))
        text = self.small_font.render(f"Wake: {agent.wakefulness:.2f}", True, text_color)
        self.screen.blit(text, (170, y))
        
        # Row 4: Drive States (if trainer has drive system)
        if hasattr(trainer, 'drive_system'):
            y = self.world_height + 10
            x_offset = 400
            
            drives = trainer.drive_system.get_drives(
                agent.energy, agent.hydration, agent.temperature
            )
            
            text = self.small_font.render("Drive States:", True, text_color)
            self.screen.blit(text, (x_offset, y))
            y += 20
            
            # Hunger
            hunger = drives.get('hunger', 0.0)
            bar_w = 150
            pygame.draw.rect(self.screen, DARK_GRAY, (x_offset, y, bar_w, 10))
            pygame.draw.rect(self.screen, ORANGE, (x_offset, y, int(bar_w * hunger), 10))
            text = self.tiny_font.render(f"Hunger: {hunger:.2f}", True, text_color)
            self.screen.blit(text, (x_offset + bar_w + 10, y - 2))
            y += 15
            
            # Thirst
            thirst = drives.get('thirst', 0.0)
            pygame.draw.rect(self.screen, DARK_GRAY, (x_offset, y, bar_w, 10))
            pygame.draw.rect(self.screen, BLUE, (x_offset, y, int(bar_w * thirst), 10))
            text = self.tiny_font.render(f"Thirst: {thirst:.2f}", True, text_color)
            self.screen.blit(text, (x_offset + bar_w + 10, y - 2))
            y += 15
            
            # Cold
            cold = drives.get('cold', 0.0)
            pygame.draw.rect(self.screen, DARK_GRAY, (x_offset, y, bar_w, 10))
            pygame.draw.rect(self.screen, CYAN, (x_offset, y, int(bar_w * cold), 10))
            text = self.tiny_font.render(f"Cold: {cold:.2f}", True, text_color)
            self.screen.blit(text, (x_offset + bar_w + 10, y - 2))
            y += 15
            
            # Curiosity
            curiosity = drives.get('curiosity', 0.0)
            pygame.draw.rect(self.screen, DARK_GRAY, (x_offset, y, bar_w, 10))
            pygame.draw.rect(self.screen, YELLOW, (x_offset, y, int(bar_w * curiosity), 10))
            text = self.tiny_font.render(f"Curiosity: {curiosity:.2f}", True, text_color)
            self.screen.blit(text, (x_offset + bar_w + 10, y - 2))
    
    def close(self):
        """Clean up pygame."""
        pygame.quit()
