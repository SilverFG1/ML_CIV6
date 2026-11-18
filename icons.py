"""Original icons for the Pygame display, drawn from shapes rather than image files.

Each icon is drawn once on a large transparent canvas, using coordinates from
0 to 1, then smooth-scaled to the sizes the display asks for. Drawing large
and scaling down softens the edges. This module never imports Pygame: the
renderer passes it in, so headless training needs no graphics packages.
"""

from functools import partial
import math

CANVAS = 128

INK = (36, 30, 42)
CREAM = (250, 243, 226)
PARCHMENT = (236, 220, 180)
WHEAT = (238, 192, 76)
WHEAT_DARK = (176, 126, 40)
SOIL = (132, 92, 52)
STEEL = (206, 214, 224)
STEEL_DARK = (122, 132, 146)
WOOD = (166, 110, 62)
WOOD_DARK = (108, 68, 36)
LEAF = (110, 182, 98)
SKY = (96, 158, 232)
ROSE = (226, 116, 154)
FLAME = (255, 150, 52)
FLAME_CORE = (255, 232, 140)
BRICK = (204, 94, 72)
STONE = (184, 178, 168)
STONE_DARK = (128, 122, 114)
GOLD = (242, 194, 62)
PURPLE = (184, 157, 207)
RED = (214, 66, 66)
TEAL = (70, 186, 196)
GLASS = (222, 240, 250)
CLEAR = (0, 0, 0, 0)

UNIT_BADGES = {
    'warrior': (228, 228, 234), 'spearman': (172, 214, 255), 'heavy_chariot': (236, 168, 236),
    'slinger': (255, 192, 134), 'archer': (255, 234, 142), 'battering_ram': (204, 204, 204),
}
MOOD_FACES = {  # face color and mouth curve: positive smiles, negative frowns
    'ecstatic': ((252, 214, 84), .16), 'happy': ((252, 214, 84), .12), 'content': ((248, 222, 130), 0),
    'displeased': ((240, 198, 120), -.06), 'unhappy': ((232, 168, 108), -.11),
    'unrest': ((226, 132, 92), -.12), 'revolt': ((214, 92, 80), -.14),
}


class Pen:
    """Draws on a square canvas with coordinates from 0 to 1 and dark outlines."""

    def __init__(self, pygame, surface):
        self.pygame, self.surface = pygame, surface
        self.n = surface.get_width()

    def _xy(self, point):
        return (round(point[0] * self.n), round(point[1] * self.n))

    def stroke(self, points, color, width=.05):
        """A line with round ends and joins, painted as a row of overlapping dots."""
        radius = max(1.0, width * self.n / 2)
        step = max(1.0, radius / 2)
        pixels = [(x * self.n, y * self.n) for x, y in points]
        for (x0, y0), (x1, y1) in zip(pixels, pixels[1:] or pixels):
            count = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for i in range(count + 1):
                t = i / count
                self.pygame.draw.circle(self.surface, color, (round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t)),
                                        round(radius))

    def line(self, points, color, width=.05, outline=True):
        if outline:
            self.stroke(points, INK, width + .05)
        self.stroke(points, color, width)

    def polygon(self, points, fill, outline=True, width=.045):
        if fill is not None:
            self.pygame.draw.polygon(self.surface, fill, [self._xy(p) for p in points])
        if outline:
            self.stroke(list(points) + [points[0]], INK, width)

    def circle(self, center, radius, fill, outline=True, width=.045):
        self.polygon(ellipse_points(center, radius, radius), fill, outline, width)

    def ellipse(self, center, rx, ry, fill, outline=True, width=.045):
        self.polygon(ellipse_points(center, rx, ry), fill, outline, width)

    def rect(self, box, fill, outline=True, width=.04, radius=0):
        x, y, w, h = box
        rect = self.pygame.Rect(round(x * self.n), round(y * self.n), round(w * self.n), round(h * self.n))
        corner = round(radius * self.n)
        if fill is not None:
            self.pygame.draw.rect(self.surface, fill, rect, border_radius=corner)
        if outline:
            self.pygame.draw.rect(self.surface, INK, rect, max(1, round(width * self.n)), border_radius=corner)

    def bar(self, start, end, half_width, fill, outline=True):
        """A straight piece with square ends, such as a handle or a blade."""
        (x0, y0), (x1, y1) = start, end
        length = math.hypot(x1 - x0, y1 - y0) or 1
        nx, ny = -(y1 - y0) / length * half_width, (x1 - x0) / length * half_width
        self.polygon([(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)], fill, outline)

    def badge(self, color):
        self.circle((.5, .5), .46, color, width=.05)


def ellipse_points(center, rx, ry, start=0.0, stop=2 * math.pi, count=56):
    cx, cy = center
    return [(cx + rx * math.cos(start + (stop - start) * i / count), cy + ry * math.sin(start + (stop - start) * i / count))
            for i in range(count + (0 if stop - start >= 2 * math.pi else 1))]


def star_points(center, outer, inner, points=5):
    cx, cy = center
    result = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + math.pi * i / points
        result.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return result


def drop_points(center, radius, height):
    """A teardrop: a round base at center with a point `height` above it."""
    cx, cy = center
    spread = math.acos(min(1.0, radius / height))
    arc = [(cx + radius * math.sin(a), cy - radius * math.cos(a))
           for a in (spread + (2 * math.pi - 2 * spread) * i / 40 for i in range(41))]
    return [(cx, cy - height)] + arc


def grain_points(center, angle, length, width):
    """A narrow lens, such as a grain of wheat, pointing along angle."""
    cx, cy = center
    ux, uy = math.cos(angle), math.sin(angle)
    side = []
    for i in range(9):
        t = i / 8
        along, across = (t - .5) * length, math.sin(math.pi * t) * width / 2
        side.append((along, across))
    outline = side + [(a, -b) for a, b in reversed(side)]
    return [(cx + a * ux - b * uy, cy + a * uy + b * ux) for a, b in outline]


def wheat_ear(pen, base, top, grains=3, scale=1.0):
    """A stalk with pairs of plump grains at its head."""
    pen.line([base, top], WHEAT_DARK, .045 * scale)
    dx, dy = top[0] - base[0], top[1] - base[1]
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    step = .13 * scale / length
    for k in range(grains):
        t = 1 - step * (k + .9)
        x, y = base[0] + dx * t, base[1] + dy * t
        for side in (-1, 1):
            spread = angle + side * .62
            offset = .055 * scale
            pen.polygon(grain_points((x + offset * math.cos(spread), y + offset * math.sin(spread)), spread,
                                     .17 * scale, .1 * scale), WHEAT, width=.035)
    pen.polygon(grain_points(top, angle, .17 * scale, .1 * scale), WHEAT, width=.035)


def sword(pen, hilt, tip, half_width=.06, blade=STEEL):
    """A sword from its pommel to its point."""
    (hx, hy), (tx, ty) = hilt, tip
    length = math.hypot(tx - hx, ty - hy)
    ux, uy = (tx - hx) / length, (ty - hy) / length
    guard = (hx + ux * .16, hy + uy * .16)
    point = (tx - ux * .1, ty - uy * .1)
    nx, ny = -uy * half_width, ux * half_width
    pen.bar(hilt, guard, .035, WOOD_DARK)
    pen.polygon([(guard[0] + nx, guard[1] + ny), (point[0] + nx, point[1] + ny), tip,
                 (point[0] - nx, point[1] - ny), (guard[0] - nx, guard[1] - ny)], blade)
    pen.bar((guard[0] - uy * .15, guard[1] + ux * .15), (guard[0] + uy * .15, guard[1] - ux * .15), .035, GOLD)
    pen.circle(hilt, .045, GOLD, width=.03)


# Yields and status.

def food(pen):
    for base, top in (((.44, .92), (.2, .3)), ((.56, .92), (.8, .3)), ((.5, .92), (.5, .14))):
        wheat_ear(pen, base, top)
    pen.bar((.36, .74), (.64, .74), .045, WOOD)


def production(pen):
    pen.bar((.22, .88), (.6, .4), .065, WOOD)
    pen.bar((.45, .2), (.83, .5), .11, STEEL)
    pen.bar((.47, .23), (.6, .33), .04, STEEL_DARK, outline=False)


def culture(pen):
    pen.rect((.24, .28, .52, .44), PARCHMENT)
    for y in (.4, .5, .6):
        pen.line([(.34, y), (.66, y)], ROSE, .035, outline=False)
    pen.rect((.12, .2, .15, .6), CREAM, radius=.07)
    pen.rect((.73, .2, .15, .6), CREAM, radius=.07)


def science(pen):
    flask = [(.41, .12), (.59, .12), (.59, .42), (.85, .86), (.15, .86), (.41, .42)]
    pen.polygon(flask, GLASS, outline=False)
    pen.polygon([(.278, .64), (.722, .64), (.85, .86), (.15, .86)], TEAL, outline=False)
    pen.polygon(flask, None)
    pen.circle((.44, .75), .035, GLASS, outline=False)
    pen.circle((.57, .7), .025, GLASS, outline=False)
    pen.bar((.36, .12), (.64, .12), .035, STEEL)


def faith(pen):
    pen.polygon(drop_points((.5, .62), .3, .54), FLAME)
    pen.polygon(drop_points((.5, .7), .15, .3), FLAME_CORE, outline=False)


def housing(pen):
    pen.rect((.24, .46, .52, .42), CREAM)
    pen.polygon([(.12, .5), (.5, .14), (.88, .5)], BRICK)
    pen.rect((.43, .62, .14, .26), WOOD)


def mood(pen, name):
    color, curve = MOOD_FACES[name]
    pen.circle((.5, .5), .41, color)
    for x in (.36, .64):
        pen.circle((x, .42), .05, INK, outline=False)
    if name in ('unrest', 'revolt'):
        pen.line([(.25, .29), (.43, .35)], INK, .045, outline=False)
        pen.line([(.75, .29), (.57, .35)], INK, .045, outline=False)
    if name == 'ecstatic':
        pen.polygon(ellipse_points((.5, .6), .2, .17, 0, math.pi, 24), INK, outline=False)
        pen.polygon(ellipse_points((.5, .7), .1, .06, 0, math.pi, 16), RED, outline=False)
        return
    mouth = [(x, .67 + curve * (1 - ((x - .5) / .17) ** 2)) for x in (.33 + .34 * i / 12 for i in range(13))]
    pen.line(mouth, INK, .05, outline=False)


def loyalty(pen):
    pen.bar((.26, .1), (.26, .92), .035, WOOD)
    pen.polygon([(.29, .14), (.86, .2), (.7, .36), (.86, .52), (.29, .5)], PURPLE)
    pen.circle((.26, .1), .055, GOLD, width=.03)


def war_weariness(pen):
    """A sword snapped in two."""
    ux, uy = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    nx, ny = -uy * .085, ux * .085
    hilt, guard, snap = (.13, .87), (.27, .73), (.5, .5)
    pen.bar(hilt, guard, .04, WOOD_DARK)
    pen.polygon([(guard[0] + nx, guard[1] + ny), (snap[0] + nx, snap[1] + ny), (snap[0] + .01, snap[1] - .05),
                 (snap[0] - .03, snap[1] - .01), (snap[0] - nx, snap[1] - ny), (guard[0] - nx, guard[1] - ny)],
                STEEL_DARK)
    pen.bar((guard[0] - uy * .17, guard[1] + ux * .17), (guard[0] + uy * .17, guard[1] - ux * .17), .04, GOLD)
    pen.circle(hilt, .05, GOLD, width=.03)
    piece = (.6, .4)
    pen.polygon([(piece[0] + nx, piece[1] + ny), (.78 + nx * .9, .22 + ny * .9), (.88, .12),
                 (.78 - nx * .9, .22 - ny * .9), (piece[0] - nx, piece[1] - ny), (piece[0] - .01, piece[1] + .04)],
                STEEL_DARK)


def governor(pen):
    pen.polygon([(.35, .55), (.25, .93), (.37, .86), (.45, .95), (.5, .62)], RED)
    pen.polygon([(.65, .55), (.75, .93), (.63, .86), (.55, .95), (.5, .62)], RED)
    pen.circle((.5, .42), .31, GOLD)
    pen.circle((.5, .42), .22, (250, 220, 112), width=.03)
    pen.polygon(star_points((.5, .43), .15, .065), CREAM, width=.03)


def pantheon(pen):
    for k in range(8):
        angle = k * math.pi / 4
        pen.polygon([(.5 + .26 * math.cos(angle - .3), .5 + .26 * math.sin(angle - .3)),
                     (.5 + .47 * math.cos(angle), .5 + .47 * math.sin(angle)),
                     (.5 + .26 * math.cos(angle + .3), .5 + .26 * math.sin(angle + .3))], GOLD, width=.035)
    pen.circle((.5, .5), .27, FLAME_CORE)


def population(pen):
    pen.polygon(ellipse_points((.5, .92), .32, .4, math.pi, 2 * math.pi, 28), (214, 204, 238))
    pen.circle((.5, .32), .17, (240, 226, 208))


def turn(pen):
    pen.polygon([(.27, .18), (.73, .18), (.53, .5), (.73, .82), (.27, .82), (.47, .5)], GLASS)
    pen.polygon([(.38, .3), (.62, .3), (.5, .46)], WHEAT, outline=False)
    pen.polygon([(.5, .62), (.68, .79), (.32, .79)], WHEAT, outline=False)
    pen.rect((.18, .1, .64, .09), WOOD, radius=.03)
    pen.rect((.18, .81, .64, .09), WOOD, radius=.03)


def city_hp(pen):
    heart = []
    for i in range(64):
        t = 2 * math.pi * i / 64
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        heart.append((.5 + x / 38, .46 - y / 38))
    pen.polygon(heart, RED)
    pen.ellipse((.34, .34), .07, .045, (240, 140, 140), outline=False)


def walls(pen):
    for x in (.12, .33, .54, .75):
        pen.rect((x, .26, .13, .2), STONE)
    pen.rect((.1, .42, .8, .46), STONE)
    for y in (.57, .72):
        pen.line([(.12, y), (.88, y)], STONE_DARK, .025, outline=False)
    for x, (top, bottom) in ((.3, (.44, .57)), (.7, (.44, .57)), (.2, (.57, .72)), (.8, (.57, .72)),
                             (.3, (.72, .86)), (.7, (.72, .86))):
        pen.line([(x, top), (x, bottom)], STONE_DARK, .025, outline=False)
    pen.polygon([(.41, .88)] + ellipse_points((.5, .66), .09, .09, math.pi, 2 * math.pi, 12) + [(.59, .88)], WOOD_DARK,
                width=.03)


def captured(pen):
    for box in ((.08, .26, .5, .3), (.42, .44, .5, .3)):
        x, y, w, h = box
        pen.rect(box, STEEL, radius=.15, width=.045)
        pen.rect((x + .1, y + .09, w - .2, h - .18), CLEAR, radius=.06, width=.035)


def settlement(pen):
    pen.line([(.5, .22), (.5, .06)], WOOD_DARK, .03)
    pen.polygon([(.5, .07), (.72, .12), (.5, .18)], PURPLE, width=.03)
    pen.polygon([(.1, .88), (.5, .2), (.9, .88)], CREAM)
    pen.polygon([(.4, .88), (.5, .56), (.6, .88)], WOOD_DARK, outline=False)


def borders(pen):
    """A tile inside a purple border, with a flag claiming it."""
    hexagon = [(.46 + .36 * math.cos(math.pi / 6 + k * math.pi / 3), .56 + .36 * math.sin(math.pi / 6 + k * math.pi / 3))
               for k in range(6)]
    pen.polygon(hexagon, (150, 192, 112), width=.04)
    pen.stroke(hexagon + [hexagon[0]], PURPLE, .05)
    pen.line([(.66, .52), (.66, .1)], WOOD_DARK, .035)
    pen.polygon([(.67, .11), (.92, .18), (.67, .27)], PURPLE, width=.03)


def farm(pen):
    pen.rect((.06, .62, .88, .3), SOIL, radius=.06)
    for y in (.71, .82):
        pen.line([(.14, y), (.86, y)], (176, 132, 82), .03, outline=False)
    for x in (.24, .5, .76):
        wheat_ear(pen, (x, .7), (x, .2), grains=2, scale=.85)


# Buildings.

def monument(pen):
    pen.rect((.22, .78, .56, .12), STONE_DARK)
    pen.polygon([(.38, .79), (.42, .26), (.5, .1), (.58, .26), (.62, .79)], STONE)
    pen.line([(.5, .3), (.5, .7)], STONE_DARK, .025, outline=False)


def granary(pen):
    pen.polygon([(.2, .88)] + ellipse_points((.5, .58), .3, .46, math.pi, 2 * math.pi, 28) + [(.8, .88)], WHEAT)
    for y, half in ((.42, .26), (.62, .3)):
        pen.line([(.5 - half, y), (.5 + half, y)], WHEAT_DARK, .025, outline=False)
    pen.polygon([(.42, .88), (.42, .72), (.5, .64), (.58, .72), (.58, .88)], WOOD_DARK, width=.03)


def library(pen):
    pen.rect((.16, .24, .17, .62), SKY)
    pen.rect((.35, .18, .17, .68), ROSE)
    pen.polygon([(.55, .86), (.71, .86), (.88, .34), (.73, .29)], LEAF)
    for x in (.16, .35):
        pen.line([(x + .04, .34), (x + .13, .34)], CREAM, .025, outline=False)
    pen.rect((.1, .86, .8, .06), WOOD)


def shrine(pen):
    pen.rect((.28, .56, .44, .32), STONE)
    pen.rect((.18, .48, .64, .1), STONE_DARK)
    pen.polygon(drop_points((.5, .38), .12, .3), FLAME, width=.035)
    pen.polygon(drop_points((.5, .41), .06, .16), FLAME_CORE, outline=False)


def barracks(pen):
    pen.polygon([(.36, .28), (.52, .07), (.8, .14), (.62, .3)], RED)
    pen.polygon(ellipse_points((.5, .6), .33, .38, math.pi, 2 * math.pi, 28), STEEL)
    pen.polygon([(.17, .58), (.83, .58), (.79, .86), (.62, .86), (.6, .68), (.4, .68), (.38, .86), (.21, .86)], STEEL)
    pen.rect((.3, .58, .4, .06), INK, outline=False)


# Districts: a colored badge with a light glyph.

def campus(pen):
    pen.badge(SKY)
    pen.polygon([(.5, .36), (.22, .29), (.22, .69), (.5, .76)], CREAM, width=.035)
    pen.polygon([(.5, .36), (.78, .29), (.78, .69), (.5, .76)], CREAM, width=.035)
    for y in (.44, .54):
        pen.line([(.29, y), (.43, y + .03)], SKY, .025, outline=False)
        pen.line([(.71, y), (.57, y + .03)], SKY, .025, outline=False)


def holy_site(pen):
    pen.badge((228, 168, 58))
    pen.polygon(star_points((.5, .52), .32, .14), CREAM, width=.035)


def encampment(pen):
    pen.badge(RED)
    shield = [(.28, .24), (.72, .24), (.72, .48)] + ellipse_points((.5, .48), .22, .32, 0, math.pi, 20)[1:] + [(.28, .48)]
    pen.polygon(shield, CREAM, width=.035)
    pen.bar((.5, .27), (.5, .76), .045, RED, outline=False)


# Units: a light badge colored by unit type, with the unit's weapon.

def warrior(pen):
    pen.badge(UNIT_BADGES['warrior'])
    sword(pen, (.24, .76), (.76, .24))


def spearman(pen):
    pen.badge(UNIT_BADGES['spearman'])
    pen.bar((.2, .8), (.66, .34), .04, WOOD)
    pen.polygon([(.6, .34), (.84, .16), (.66, .4)], STEEL, width=.035)


def heavy_chariot(pen):
    pen.badge(UNIT_BADGES['heavy_chariot'])
    pen.circle((.5, .5), .3, WOOD)
    pen.circle((.5, .5), .21, UNIT_BADGES['heavy_chariot'], width=.035)
    for k in range(6):
        angle = k * math.pi / 3
        pen.line([(.5, .5), (.5 + .22 * math.cos(angle), .5 + .22 * math.sin(angle))], WOOD_DARK, .035, outline=False)
    pen.circle((.5, .5), .07, STEEL, width=.03)


def slinger(pen):
    """A stone whirled around on a cord."""
    pen.badge(UNIT_BADGES['slinger'])
    whirl = ellipse_points((.5, .5), .26, .26, 1.2, 1.2 + 4.4, 28)
    pen.line(whirl, WOOD_DARK, .04)
    pen.circle((.5, .5), .05, WOOD, width=.03)
    end = whirl[-1]
    pen.circle(end, .11, STONE_DARK, width=.035)
    for radius in (.36, .4):
        pen.stroke(ellipse_points((.5, .5), radius, radius, 5.9, 6.6, 6), INK, .025)


def archer(pen):
    """A drawn bow with an arrow pointing right."""
    pen.badge(UNIT_BADGES['archer'])
    bow = ellipse_points((.3, .5), .38, .38, -.95, .95, 20)
    pen.line([bow[0], (.3, .5), bow[-1]], CREAM, .02, outline=False)
    pen.line(bow, WOOD, .06)
    pen.line([(.24, .5), (.78, .5)], WOOD_DARK, .035)
    pen.polygon([(.74, .42), (.9, .5), (.74, .58)], STEEL, width=.03)
    pen.polygon([(.2, .43), (.31, .5), (.2, .57), (.15, .5)], RED, width=.025)


def battering_ram(pen):
    pen.badge(UNIT_BADGES['battering_ram'])
    pen.bar((.16, .46), (.74, .46), .09, WOOD)
    pen.polygon([(.72, .34), (.86, .4), (.86, .52), (.72, .58)], STEEL, width=.035)
    for x in (.3, .6):
        pen.circle((x, .68), .1, WOOD_DARK, width=.035)
        pen.circle((x, .68), .03, STEEL, outline=False)


# Civilians: the city's purple badge with their tools.

def builder(pen):
    pen.badge(PURPLE)
    pen.bar((.28, .8), (.6, .36), .045, WOOD)
    head = (ellipse_points((.6, .62), .36, .36, 3.7, 5.7, 16)
            + ellipse_points((.6, .62), .28, .28, 5.7, 3.7, 16))
    pen.polygon(head, STEEL, width=.035)


def settler(pen):
    pen.badge(PURPLE)
    pen.polygon([(.18, .58)] + ellipse_points((.52, .58), .34, .3, math.pi, 2 * math.pi, 24) + [(.86, .58)], CREAM,
                width=.035)
    for x in (.4, .64):
        pen.line([(x, .32), (x, .58)], PARCHMENT, .02, outline=False)
    pen.rect((.16, .56, .72, .12), WOOD, width=.03)
    for x in (.32, .7):
        pen.circle((x, .74), .1, WOOD_DARK, width=.035)
        pen.circle((x, .74), .03, STEEL, outline=False)


ICONS = {
    'food': food, 'production': production, 'culture': culture, 'science': science, 'faith': faith,
    'housing': housing, 'loyalty': loyalty, 'war_weariness': war_weariness, 'governor': governor,
    'pantheon': pantheon, 'population': population, 'turn': turn, 'city_hp': city_hp, 'walls': walls,
    'captured': captured, 'settlement': settlement, 'farm': farm, 'borders': borders,
    **{f'mood_{name}': partial(mood, name=name) for name in MOOD_FACES},
    **{f'building_{name}': draw for name, draw in (
        ('monument', monument), ('granary', granary), ('ancient_walls', walls), ('library', library),
        ('shrine', shrine), ('barracks', barracks))},
    'district_campus': campus, 'district_holy_site': holy_site, 'district_encampment': encampment,
    **{f'unit_{name}': draw for name, draw in (
        ('warrior', warrior), ('spearman', spearman), ('heavy_chariot', heavy_chariot), ('slinger', slinger),
        ('archer', archer), ('battering_ram', battering_ram))},
    'civilian_builder': builder, 'civilian_settler': settler,
}


class IconSet:
    """Draws icons on first use and caches them by name and size."""

    def __init__(self, pygame):
        self.pygame = pygame
        self._canvases = {}
        self._sized = {}

    def get(self, name, size):
        size = max(4, int(size))
        key = (name, size)
        if key not in self._sized:
            if name not in self._canvases:
                if name not in ICONS:
                    raise KeyError(f'no icon named {name!r}')
                canvas = self.pygame.Surface((CANVAS, CANVAS), self.pygame.SRCALPHA)
                # Transparent pixels carry the outline color, so scaling does not leave light fringes.
                canvas.fill(INK + (0,))
                ICONS[name](Pen(self.pygame, canvas))
                self._canvases[name] = canvas
            self._sized[key] = self.pygame.transform.smoothscale(self._canvases[name], (size, size))
        return self._sized[key]
