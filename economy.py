"""A Civ VI-style economy for the defending city: growth, production and borders.

Each turn the city collects yields from its centre tile, the tiles its citizens
work and its buildings. Food feeds and grows the population within its
housing, production completes a queue of buildings, and culture claims new
tiles. Formulas follow Civ VI's base game; the README lists the
simplifications.
"""

from class_hex import YIELD_TYPES, hex_distance, neighbor_positions

FOOD_PER_CITIZEN = 2
# A city centre always yields at least this much.
CITY_CENTER_MINIMUM = (('food', 2), ('production', 1))
# Citizens work, and borders grow to, tiles up to this far from the centre.
WORK_RADIUS = 3
# How much the city values each yield when picking tiles to work or claim.
TILE_WEIGHTS = {'food': 2, 'production': 2, 'gold': 1, 'science': 1, 'culture': 1, 'faith': 1}


def growth_threshold(population):
    """Food needed to grow from `population` to the next size: 15, 24, 33, 43, ..."""
    n = population - 1
    return int(15 + 8 * n + n ** 1.5)


def growth_multiplier(population, housing):
    """Growth slows near the housing limit and stops five citizens above it."""
    if population <= housing - 2:
        return 1.0
    if population == housing - 1:
        return 0.5
    if population <= housing + 4:
        return 0.25
    return 0.0


def border_cost(tiles_acquired):
    """Culture needed for the next tile: 10, then 10 + (6 x tiles so far) ^ 1.3."""
    return 10 + (6 * tiles_acquired) ** 1.3


def tile_score(yields):
    return sum(TILE_WEIGHTS[kind] * amount for kind, amount in yields.items())


def _add(totals, yields):
    for kind, amount in yields.items():
        totals[kind] = totals.get(kind, 0) + amount


class CityEconomy:
    """The city's stored food, production and culture, its territory and population.

    `env` is the Game; `complete_build(name)` and `is_built(name)` on it apply
    and check finished items, and `build_costs` maps each item to its cost.
    """

    def __init__(self, env, population=1, build_queue=(), build_costs=None):
        self.env = env
        self.population = population
        self.food = self.production = self.culture = 0.0
        self.tiles_acquired = 0
        self.build_queue = tuple(build_queue)
        self.build_costs = dict(build_costs or {})
        center = env.city.position
        # A new city owns its centre and the six tiles around it.
        self.territory = {center} | {p for p in neighbor_positions(center) if p in env.map.grid}

    @property
    def _center(self):
        return self.env.city.position

    def _rank(self, positions):
        """Best tiles first: highest score, then most food, then nearest."""
        grid = self.env.map.grid
        return sorted(positions, key=lambda p: (-tile_score(grid[p].yields), -grid[p].yields.get('food', 0),
                                                hex_distance(p, self._center), p))

    def workable_tiles(self):
        """Owned tiles a citizen could work: not the centre, not impassable, not occupied by a unit."""
        occupied = {unit.position for unit in self.env.units if unit.alive}
        grid = self.env.map.grid
        return [p for p in self.territory
                if p != self._center and not grid[p].spec.impassable and p not in occupied]

    def worked_tiles(self):
        """The best workable tiles, one per citizen, trading for food when the city would starve."""
        grid = self.env.map.grid
        ranked = self._rank(self.workable_tiles())
        worked, spare = ranked[:self.population], ranked[self.population:]

        def food(position):
            return grid[position].yields.get('food', 0)

        center_food = self.center_yields().get('food', 0)
        building_food = self.env.city.building_yields.get('food', 0)
        while spare and center_food + building_food + sum(map(food, worked)) < FOOD_PER_CITIZEN * self.population:
            poorest = min(worked, key=lambda p: (food(p), -worked.index(p)))
            richest = max(spare, key=lambda p: (food(p), -spare.index(p)))
            if food(richest) <= food(poorest):
                break  # No swap adds food.
            worked[worked.index(poorest)] = richest
            spare.remove(richest)
        return worked

    def center_yields(self):
        totals = dict(self.env.map.grid[self._center].yields)
        for kind, minimum in CITY_CENTER_MINIMUM:
            totals[kind] = max(totals.get(kind, 0), minimum)
        return totals

    def yields(self):
        """This turn's yields from the centre, worked tiles and buildings."""
        totals = self.center_yields()
        for position in self.worked_tiles():
            _add(totals, self.env.map.grid[position].yields)
        _add(totals, self.env.city.building_yields)
        return {kind: totals[kind] for kind in YIELD_TYPES if totals.get(kind)}

    def base_housing(self):
        """2, or 3 next to water, or 5 next to a river (fresh water)."""
        grid = self.env.map.grid
        neighbors = [p for p in neighbor_positions(self._center) if p in grid]
        if any(self.env.map.has_river(self._center, p) for p in neighbors):
            return 5
        if any(grid[p].is_water for p in neighbors):
            return 3
        return 2

    def housing(self):
        return self.base_housing() + self.env.city.building_housing

    def food_surplus(self, yields=None):
        yields = self.yields() if yields is None else yields
        return yields.get('food', 0) - FOOD_PER_CITIZEN * self.population

    def current_build(self):
        return next((name for name in self.build_queue if not self.env.is_built(name)), None)

    def border_candidates(self):
        """Unowned tiles next to the territory, within reach, best first."""
        grid = self.env.map.grid
        candidates = {q for p in self.territory for q in neighbor_positions(p)
                      if q in grid and q not in self.territory and hex_distance(q, self._center) <= WORK_RADIUS}
        return self._rank(candidates)

    def turn(self):
        """Collect this turn's yields, then grow, build and expand."""
        yields = self.yields()
        self._grow(yields)
        self._build(yields.get('production', 0))
        self._expand(yields.get('culture', 0))

    def _grow(self, yields):
        city = self.env.city
        surplus = self.food_surplus(yields)
        if surplus > 0:
            surplus *= growth_multiplier(self.population, self.housing())
        self.food += surplus
        if self.food < 0:
            self.food = 0.0
            if self.population > 1:
                self.population -= 1
                city._event('city_starved', population=self.population)
        elif self.food >= growth_threshold(self.population):
            self.food -= growth_threshold(self.population)
            self.population += 1
            city._event('city_grew', population=self.population)

    def _build(self, production):
        item = self.current_build()
        if item is None:
            return
        self.production += production
        cost = self.build_costs[item]
        if self.production >= cost:
            self.production -= cost
            self.env.complete_build(item)
            self.env.city._event('building_completed', building=item)
            if self.current_build() is None:
                self.production = 0.0  # Nothing left to build.

    def _expand(self, culture):
        candidates = self.border_candidates()
        if not candidates:
            return
        self.culture += culture
        cost = border_cost(self.tiles_acquired)
        if self.culture >= cost:
            self.culture -= cost
            self.territory.add(candidates[0])
            self.tiles_acquired += 1
            self.env.city._event('border_expanded', tile=list(candidates[0]))

    def observation(self, build_items):
        """City-level features: size, progress towards growth, the next item and the next tile."""
        item = self.current_build()
        progress = self.production / self.build_costs[item] if item else 0.0
        return [self.population / 20, min(1.0, self.food / growth_threshold(self.population)),
                min(1.0, progress), min(1.0, self.culture / border_cost(self.tiles_acquired)),
                self.housing() / 20, *(float(item == name) for name in build_items)]

    def status_text(self):
        yields = self.yields()
        item = self.current_build()
        building = (f'{item.replace("_", " ").title()} {self.production:.0f}/{self.build_costs[item]}'
                    if item else 'Nothing to build')
        return (f'Pop {self.population}   Housing {self.housing()}   '
                f'Food {self.food:.0f}/{growth_threshold(self.population)} ({self.food_surplus(yields):+d})   '
                f'{building} (+{yields.get("production", 0)})   '
                f'Culture {self.culture:.0f}/{border_cost(self.tiles_acquired):.0f} (+{yields.get("culture", 0)})')
