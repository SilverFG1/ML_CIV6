"""A Civ VI-style economy for the defending city: growth, production and borders.

Each turn the city collects yields from its centre tile, the tiles its citizens
work, its buildings and its districts. Food feeds and grows the population
within its housing, production completes a queue of districts and buildings,
culture claims new tiles, and faith founds a pantheon. Amenities set the
city's mood, war weariness costs amenities, and loyalty can make the city
declare independence. Formulas follow Civ VI (Gathering Storm where the
expansions changed them); the README lists the simplifications.
"""

from dataclasses import dataclass

from class_hex import YIELD_TYPES, hex_distance, neighbor_positions

FOOD_PER_CITIZEN = 2
# A city centre always yields at least this much.
CITY_CENTER_MINIMUM = (('food', 2), ('production', 1))
# Citizens work, and districts go on, owned tiles up to this far from the centre.
WORK_RADIUS = 3
# Culture claims tiles out to this far, ring by ring.
BORDER_RADIUS = 5
# How much the city values each yield when picking tiles to work or claim.
TILE_WEIGHTS = {'food': 2, 'production': 2, 'gold': 1, 'science': 1, 'culture': 1, 'faith': 1}
MAX_CITY_POPULATION = 20

# A farm adds 1 food and half a house. Builders farm flat grassland and plains;
# Civ VI does not allow farms on woods.
FARM_FOOD = 1
FARM_HOUSING = 0.5
FARM_LAND = ('GRASSLAND', 'PLAINS')

# Amenities (Civ VI): each luxury resource provides one. The balance sets the
# city's mood, which changes its growth and its non-food yields. These are
# Gathering Storm's tiers, which raised the happy and ecstatic thresholds to
# +3 and +5; sources differ on some yield changes.
AMENITY_MOODS = (  # (lowest balance, mood, growth change, non-food yield change)
    (5, 'ecstatic', .20, .20),
    (3, 'happy', .10, .10),
    (0, 'content', 0, 0),
    (-2, 'displeased', -.15, -.05),
    (-4, 'unhappy', -.30, -.10),
    (-6, 'unrest', -1.0, -.15),
    (None, 'revolt', -1.0, -.20),
)
MAX_LUXURIES = 8

# War weariness (Gathering Storm global parameters). Each combat costs both
# sides 16 points in the ancient era. The defender fights on its own land and
# has no units to lose, so neither multiplier applies. While at war, weariness
# fades by 50 points a turn. Every 400 points cost the city one amenity, but
# never more amenities than the city needs.
WAR_WEARINESS_PER_COMBAT = 16
WAR_WEARINESS_DECAY = 50
WAR_WEARINESS_PER_AMENITY = 400
MAX_WAR_WEARINESS = 10000  # largest starting value a scenario accepts

# Loyalty (Rise and Fall and Gathering Storm), 0-100. Each citizen within 9
# tiles adds 1 loyalty a turn, 10% less per tile away; capital citizens count
# double. Domestic minus foreign pressure is capped at 20 either way, and the
# city's mood adds or removes loyalty. At 0 the city declares independence.
LOYALTY_RANGE = 9
MAX_PRESSURE = 20
CAPITAL_PRESSURE = 2
LOYALTY_FROM_MOOD = {'ecstatic': 6, 'happy': 3, 'content': 0, 'displeased': -3,
                     'unhappy': -6, 'unrest': -6, 'revolt': -6}
# A city of the attacker's civilization must be at least this far away.
MIN_FOREIGN_CITY_DISTANCE = 4


@dataclass(frozen=True)
class GovernorSpec:
    title: str  # the governor's first title, which comes with the appointment
    establish_turns: int = 5
    promotions: tuple = ()  # the first-tier promotions this simulation offers


# Governors. The ancient era's two governor titles appoint one governor and give
# it one promotion. The governor arrives when the game starts and takes effect
# once established: after 5 turns, or 3 for Victor. An established governor adds
# 8 loyalty a turn. Promotions are offered only where they can matter here.
GOVERNORS = {
    'victor': GovernorSpec('Redoubt', 3, ('garrison_commander', 'defense_logistics')),
    'liang': GovernorSpec('Guildmaster', promotions=('zoning_commissioner', 'reinforced_materials')),
    'magnus': GovernorSpec('Groundbreaker', promotions=('provision', 'surplus_logistics')),
    'pingala': GovernorSpec('Librarian', promotions=('connoisseur', 'researcher')),
    'reyna': GovernorSpec('Land Acquisition'),
    'amani': GovernorSpec('Messenger'),
    'moksha': GovernorSpec('Bishop'),
}
GOVERNOR_PROMOTIONS = tuple(sorted({name for spec in GOVERNORS.values() for name in spec.promotions}))
GOVERNOR_LOYALTY = 8
LIBRARIAN_BONUS = .15  # Pingala: +15% science and culture
SURPLUS_LOGISTICS_FOOD = .20  # Magnus: +20% food
ZONING_COMMISSIONER_BONUS = .20  # Liang: +20% production toward districts

# Religion in the ancient era is the pantheon, founded once the civilization
# stores 25 faith. Only beliefs that can matter here are offered: Fertility
# Rites (+10% growth, and a free builder as in Gathering Storm), City Patron
# Goddess (+25% production toward districts while the city has no specialty
# district) and Religious Settlements (borders grow 15% faster, and a free
# settler as in Gathering Storm).
PANTHEON_COST = 25
PANTHEONS = ('fertility_rites', 'city_patron_goddess', 'religious_settlements')
PANTHEON_CIVILIANS = {'fertility_rites': 'builder', 'religious_settlements': 'settler'}
FERTILITY_RITES_GROWTH = .10
CITY_PATRON_GODDESS_BONUS = .25
RELIGIOUS_SETTLEMENTS_BORDERS = .15
MAX_STARTING_FAITH = 500


@dataclass(frozen=True)
class BuildingSpec:
    """A Civ VI building: production cost, yields, housing and loyalty, and where it stands."""
    civ6_type: str
    cost: int
    yields: tuple = ()  # (yield type, amount) pairs
    housing: float = 0
    technology: str | None = None
    district: str = 'city_center'
    loyalty: float = 0


@dataclass(frozen=True)
class DistrictSpec:
    """An ancient-era district and its adjacency bonus."""
    civ6_type: str
    technology: str
    cost: int = 54  # base cost; Civ VI raises it with tech and civic progress
    yield_type: str | None = None  # what its adjacency bonus produces
    mountain: int = 0  # adjacency per adjacent mountain
    woods_pair: int = 0  # adjacency per two adjacent woods (forests)
    district_pair: int = 0  # adjacency per two adjacent districts (the city centre counts)
    min_distance: int = 1  # closest it may be to the city centre


# Base-game costs from community references; some wikis list 50 for the Monument
# and Granary. A Monument also adds 1 loyalty a turn. The Water Mill and the
# Government Plaza's buildings are left out: the first needs a river beside the
# city centre in a way this map does not model, the others Classical governments.
BUILDING_SPECS = {
    'monument': BuildingSpec('BUILDING_MONUMENT', cost=60, yields=(('culture', 2),), loyalty=1),
    'granary': BuildingSpec('BUILDING_GRANARY', cost=65, yields=(('food', 1),), housing=2,
                            technology='TECH_POTTERY'),
    'library': BuildingSpec('BUILDING_LIBRARY', cost=80, yields=(('science', 2),), technology='TECH_WRITING',
                            district='campus'),
    'shrine': BuildingSpec('BUILDING_SHRINE', cost=65, yields=(('faith', 2),), technology='TECH_ASTROLOGY',
                           district='holy_site'),
    'barracks': BuildingSpec('BUILDING_BARRACKS', cost=80, yields=(('production', 1),), housing=1,
                             technology='TECH_BRONZE_WORKING', district='encampment'),
}
BUILDING_TYPES = tuple(BUILDING_SPECS)

# Districts go on owned land within 3 hexes of the city centre; the Encampment
# cannot be next to it, blocks movement and strikes nearby units.
DISTRICT_SPECS = {
    'campus': DistrictSpec('DISTRICT_CAMPUS', 'TECH_WRITING', yield_type='science', mountain=1, district_pair=1),
    'holy_site': DistrictSpec('DISTRICT_HOLY_SITE', 'TECH_ASTROLOGY', yield_type='faith', mountain=1,
                              woods_pair=1, district_pair=1),
    'encampment': DistrictSpec('DISTRICT_ENCAMPMENT', 'TECH_BRONZE_WORKING', min_distance=2),
}
DISTRICT_TYPES = tuple(DISTRICT_SPECS)


def growth_threshold(population):
    """Food needed to grow from `population` to the next size: 15, 24, 33, 43, ..."""
    n = population - 1
    return int(15 + 8 * n + n ** 1.5)


def growth_multiplier(population, housing):
    """Growth slows near the housing limit and stops five citizens above it.

    Full rate more than one citizen below housing, half within one citizen of it, a quarter at or above it and
    none five or more above it. Housing may be fractional (empire mode's improvements add half a house each).
    """
    if population >= housing + 5:
        return 0.0
    if population >= housing:
        return 0.25
    if population >= housing - 1:
        return 0.5
    return 1.0


def border_cost(tiles_acquired):
    """Culture needed for the next tile: 10, then 10 + (6 x tiles so far) ^ 1.3."""
    return 10 + (6 * tiles_acquired) ** 1.3


def tile_score(yields):
    return sum(TILE_WEIGHTS[kind] * amount for kind, amount in yields.items())


def amenities_needed(population):
    """Amenities a capital needs: none for its first two citizens, one for the third, then one per two.

    Gathering Storm gives capitals one free amenity; other cities would need one more.
    """
    return max(0, (population - 1) // 2)


def amenity_mood(balance):
    """Return (mood, growth change, non-food yield change) for an amenity balance."""
    for lowest, mood, growth, yields in AMENITY_MOODS:
        if lowest is None or balance >= lowest:
            return mood, growth, yields


def loyalty_effects(loyalty):
    """Return (growth change, yield change): -25% below 75 loyalty, and -75% growth, -50% yields below 25."""
    if loyalty >= 75:
        return 0, 0
    if loyalty >= 25:
        return -.25, -.25
    return -.75, -.5


def citizen_pressure(population, distance, capital=False):
    """Loyalty a turn from a city's citizens at this distance."""
    if distance > LOYALTY_RANGE:
        return 0
    return population * (10 - distance) / 10 * (CAPITAL_PRESSURE if capital else 1)


def district_limit(population):
    """Specialty districts a city of this size may have: one, plus one per three citizens beyond the first."""
    return 1 + (population - 1) // 3


def _add(totals, yields):
    for kind, amount in yields.items():
        totals[kind] = totals.get(kind, 0) + amount


class CityEconomy:
    """The city's stored food, production, culture and faith, its territory, population and mood.

    `env` is the Game. `complete_build(name)` and `is_built(name)` on it apply
    and check finished items, `build_costs` maps each item to its cost, and
    `config` (a GameConfig) supplies the starting population, luxuries, war
    weariness, loyalty, foreign city, governor, pantheon and faith.
    """

    def __init__(self, env, config, build_queue=(), build_costs=None):
        self.env = env
        self.config = config
        self.population = config.city_population
        self.food = self.culture = 0.0
        self.progress = {}  # production stored toward each item
        self.tiles_acquired = 0
        self.build_queue = tuple(build_queue)
        self.build_costs = dict(build_costs or {})
        self.completed = []
        self.districts = {}  # district name -> position
        self.farms = set()
        self.founded_cities = []  # positions of cities the settlers founded
        self.war_weariness = config.city_war_weariness
        self.loyalty = float(config.city_loyalty)
        self.independent = False
        self.governor_turns = 0
        self.faith = float(config.city_faith)
        self.pantheon_founded = False
        center = env.city.position
        # A new city owns its centre and the six tiles around it.
        self.territory = {center} | {p for p in neighbor_positions(center) if p in env.map.grid}

    @property
    def _center(self):
        return self.env.city.position

    # Production toward the item being built; settable so a scenario can start partway.
    @property
    def production(self):
        item = self.current_build()
        return self.progress.get(item, 0.0) if item else 0.0

    @production.setter
    def production(self, amount):
        item = self.current_build()
        if item is not None:
            self.progress[item] = float(amount)

    def tile_yields(self, position):
        """A tile's yields, including a farm."""
        yields = dict(self.env.map.grid[position].yields)
        if position in self.farms:
            yields['food'] = yields.get('food', 0) + FARM_FOOD
        return yields

    def _rank(self, positions):
        """Best tiles first: highest score, then most food, then nearest."""
        return sorted(positions, key=lambda p: (-tile_score(self.tile_yields(p)), -self.tile_yields(p).get('food', 0),
                                                hex_distance(p, self._center), p))

    def _towns(self):
        """Tiles of the other cities on the map, which the city can never own."""
        env = self.env
        towns = {city.position for city in env.cities if city is not env.city}
        return towns | {env.capital.position} if env.capital is not None else towns

    def territory_tiles(self):
        """Owned tiles within reach of the centre that are not the centre, impassable or a district."""
        grid = self.env.map.grid
        return [p for p in self.territory
                if p != self._center and hex_distance(p, self._center) <= WORK_RADIUS
                and not grid[p].spec.impassable and grid[p].district is None]

    def workable_tiles(self):
        """Owned tiles a citizen could work: not the centre, impassable, a district or occupied by a unit."""
        occupied = {unit.position for unit in self.env.units if unit.alive}
        return [p for p in self.territory_tiles() if p not in occupied]

    def blocked_tiles(self):
        """Owned tiles a citizen could otherwise work that one of the squad's units stands on."""
        occupied = {unit.position for unit in self.env.units if unit.alive}
        return sorted((p for p in self.territory_tiles() if p in occupied), key=lambda p: (p[1], p[0]))

    def worked_tiles(self):
        """The best workable tiles, one per citizen, trading for food when the city would starve."""
        ranked = self._rank(self.workable_tiles())
        worked, spare = ranked[:self.population], ranked[self.population:]

        def food(position):
            return self.tile_yields(position).get('food', 0)

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

    def adjacency(self, district, position):
        """A district's Civ VI adjacency bonus at position; the city centre counts as a district."""
        spec = DISTRICT_SPECS[district]
        neighbors = self.env.map.grid[position].get_neighbors(self.env.map.grid)
        mountains = sum(cell.spec.elevation == 'MOUNTAIN' for cell in neighbors)
        woods = sum(cell.feature == 'FOREST' and cell.district is None for cell in neighbors)
        districts = sum(cell.district is not None or cell.index == self._center for cell in neighbors)
        return spec.mountain * mountains + spec.woods_pair * (woods // 2) + spec.district_pair * (districts // 2)

    def district_yields(self):
        totals = {}
        for district, position in self.districts.items():
            kind = DISTRICT_SPECS[district].yield_type
            if kind:
                totals[kind] = totals.get(kind, 0) + self.adjacency(district, position)
        return totals

    def yields(self):
        """This turn's yields from the centre, worked tiles, buildings, districts and governor.

        Mood and loyalty scale every yield but food; Magnus can raise food.
        """
        totals = self.center_yields()
        for position in self.worked_tiles():
            _add(totals, self.tile_yields(position))
        _add(totals, self.env.city.building_yields)
        _add(totals, self.district_yields())
        if self.governor_has('connoisseur'):
            totals['culture'] = totals.get('culture', 0) + self.population
        if self.governor_has('researcher'):
            totals['science'] = totals.get('science', 0) + self.population
        if self.governor_has('surplus_logistics') and totals.get('food'):
            totals['food'] = round(totals['food'] * (1 + SURPLUS_LOGISTICS_FOOD), 2)
        librarian = LIBRARIAN_BONUS if self.governor_has('pingala') else 0
        change = self.mood()[2] + loyalty_effects(self.loyalty)[1]
        for kind in totals:
            if kind != 'food':
                bonus = change + (librarian if kind in ('culture', 'science') else 0)
                if bonus:
                    totals[kind] = round(totals[kind] * max(0, 1 + bonus), 2)
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
        return self.base_housing() + self.env.city.building_housing + FARM_HOUSING * len(self.farms)

    def food_surplus(self, yields=None):
        yields = self.yields() if yields is None else yields
        return yields.get('food', 0) - FOOD_PER_CITIZEN * self.population

    def growth_surplus(self, yields=None):
        """Food stored this turn: mood, loyalty, Fertility Rites and housing scale a surplus but never a shortfall."""
        surplus = self.food_surplus(yields)
        if surplus > 0:
            change = self.mood()[1] + loyalty_effects(self.loyalty)[0]
            if self.pantheon_has('fertility_rites'):
                change += FERTILITY_RITES_GROWTH
            rate = max(0, 1 + change) * growth_multiplier(self.population, self.housing())
            if rate != 1:
                surplus = round(surplus * rate, 4)
        return surplus

    # Amenities and mood.
    def amenities_lost_to_war(self):
        return min(self.war_weariness // WAR_WEARINESS_PER_AMENITY, amenities_needed(self.population))

    def amenities(self):
        """Return (amenities available, amenities needed)."""
        return self.config.city_luxuries - self.amenities_lost_to_war(), amenities_needed(self.population)

    def mood(self):
        """The city's mood and its growth and non-food yield changes."""
        available, needed = self.amenities()
        return amenity_mood(available - needed)

    def combat_fought(self):
        """Each combat around the city adds to its civilization's war weariness."""
        self.war_weariness += WAR_WEARINESS_PER_COMBAT

    # Governor and pantheon.
    def governor_established(self):
        governor = self.config.city_governor
        return bool(governor) and self.governor_turns >= GOVERNORS[governor].establish_turns

    def governor_has(self, effect):
        """Whether the established governor has this title: its first title (its name) or its promotion."""
        if not self.governor_established():
            return False
        return effect in (self.config.city_governor, self.config.city_governor_promotion)

    def pantheon_has(self, belief):
        return self.pantheon_founded and self.config.city_pantheon == belief

    # Loyalty.
    def loyalty_sources(self):
        """This turn's loyalty change from each source."""
        domestic = citizen_pressure(self.population, 0, capital=True)
        domestic += sum(citizen_pressure(1, hex_distance(p, self._center)) for p in self.founded_cities)
        foreign = citizen_pressure(*self.config.foreign_city) if self.config.foreign_city else 0
        sources = {'citizen_pressure': round(max(-MAX_PRESSURE, min(MAX_PRESSURE, domestic - foreign)), 2),
                   'amenities': LOYALTY_FROM_MOOD[self.mood()[0]]}
        buildings = self.env.city.building_loyalty
        if buildings:
            sources['buildings'] = buildings
        if self.governor_established():
            sources['governor'] = GOVERNOR_LOYALTY
        return sources

    def loyalty_per_turn(self):
        return round(sum(self.loyalty_sources().values()), 2)

    # Districts and the build queue.
    def district_site(self, district):
        """The free owned tile with the best adjacency, then the lowest yields, or None.

        An encampment cannot stand next to the centre, and it never cuts any
        tile off from the city, since it blocks movement.
        """
        env = self.env
        occupied = ({u.position for u in env.units if u.alive} | {c.position for c in env.civilians if c.alive}
                    | {r.position for r in env.raiders if r.alive})
        spec = DISTRICT_SPECS[district]
        sites = [p for p in self.territory_tiles()
                 if hex_distance(p, self._center) >= spec.min_distance and p not in occupied
                 and not env.map.grid[p].is_water]
        if district == 'encampment':
            sites = [p for p in sites if not self._cuts_off(p)]
        if not sites:
            return None
        return max(sites, key=lambda p: (self.adjacency(district, p), -tile_score(self.tile_yields(p)), -p[1], -p[0]))

    def _cuts_off(self, position):
        """Whether blocking position would separate any open tile from the city."""
        grid = self.env.map.grid
        before = self.env.map.reachable(self._center) - {position}
        reached, pending = {self._center}, [self._center]
        while pending:
            current = pending.pop()
            for cell in grid[current].get_neighbors(grid):
                if cell.index not in reached and cell.index != position and not cell.block_path:
                    reached.add(cell.index)
                    pending.append(cell.index)
        return before != reached

    def can_build(self, item):
        if self.env.is_built(item):
            return False
        if item in DISTRICT_SPECS:
            return len(self.districts) < district_limit(self.population) and self.district_site(item) is not None
        if item in BUILDING_SPECS:
            district = BUILDING_SPECS[item].district
            return district == 'city_center' or district in self.districts
        return True

    def current_build(self):
        """The first queued item the city does not have and can work on now."""
        return next((name for name in self.build_queue if self.can_build(name)), None)

    def place_district(self, district):
        """Put a district on its site; return the site, or None when there is none."""
        site = self.district_site(district)
        if site is None:
            return None
        cell = self.env.map.grid[site]
        cell.district, cell.fortified = district, district == 'encampment'
        self.farms.discard(site)
        self.districts[district] = site
        return site

    # Borders.
    def border_candidates(self):
        """Unowned tiles next to the territory, ring by ring out to BORDER_RADIUS; workable land with the best
        yields first within a ring."""
        grid = self.env.map.grid
        towns = self._towns()
        candidates = {q for p in self.territory for q in neighbor_positions(p)
                      if q in grid and q not in self.territory and q not in towns
                      and hex_distance(q, self._center) <= BORDER_RADIUS}
        return sorted(candidates, key=lambda p: (hex_distance(p, self._center), grid[p].spec.impassable,
                                                 -tile_score(self.tile_yields(p)),
                                                 -self.tile_yields(p).get('food', 0), p))

    def claim_tile(self):
        """Add the next tile to the territory without spending culture; return it, or None."""
        candidates = self.border_candidates()
        if not candidates:
            return None
        self.territory.add(candidates[0])
        self.tiles_acquired += 1
        return candidates[0]

    def border_growth(self, yields=None):
        """Culture a turn that goes toward new tiles."""
        culture = (self.yields() if yields is None else yields).get('culture', 0)
        if self.pantheon_has('religious_settlements'):
            culture *= 1 + RELIGIOUS_SETTLEMENTS_BORDERS
        return round(culture, 2)

    # The turn.
    def turn(self):
        """Collect this turn's yields, then grow, build, expand, gather faith and update loyalty."""
        if self.config.city_governor:
            self.governor_turns += 1
        self.war_weariness = max(0, self.war_weariness - WAR_WEARINESS_DECAY)
        yields = self.yields()
        self._grow(yields)
        self._build(yields.get('production', 0))
        self._expand(self.border_growth(yields))
        if self.config.city_pantheon:
            self._gather_faith(yields.get('faith', 0))
        self._update_loyalty()

    def _grow(self, yields):
        city = self.env.city
        self.food = round(self.food + self.growth_surplus(yields), 4)
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
        """Put production into the first item the city can build; overflow goes to the next one.

        Production is lost when nothing in the queue can be built now; partial
        progress on a waiting item is kept.
        """
        while production > 0:
            item = self.current_build()
            if item is None:
                return
            rate = 1
            if item in DISTRICT_SPECS:
                # Liang's Zoning Commissioner and City Patron Goddess speed up districts.
                rate += ZONING_COMMISSIONER_BONUS if self.governor_has('zoning_commissioner') else 0
                rate += (CITY_PATRON_GODDESS_BONUS
                         if self.pantheon_has('city_patron_goddess') and not self.districts else 0)
            cost = self.build_costs[item]
            needed = cost - self.progress.get(item, 0.0)
            if production * rate < needed:
                self.progress[item] = round(self.progress.get(item, 0.0) + production * rate, 4)
                return
            self.progress[item] = float(cost)
            production = round(production - needed / rate, 4)
            self.env.complete_build(item)
            self.completed.append(item)
            kind = 'district' if item in DISTRICT_SPECS else 'building'
            self.env.city._event(f'{kind}_completed', **{kind: item})

    def _expand(self, culture):
        candidates = self.border_candidates()
        if not candidates:
            return
        self.culture = round(self.culture + culture, 4)
        cost = border_cost(self.tiles_acquired)
        if self.culture >= cost:
            self.culture -= cost
            tile = self.claim_tile()
            self.env.city._event('border_expanded', tile=list(tile))

    def _gather_faith(self, faith):
        """Store the city's faith, and found the pantheon once there is enough."""
        self.faith = round(self.faith + faith, 2)
        if not self.pantheon_founded and self.faith >= PANTHEON_COST:
            self.faith = round(self.faith - PANTHEON_COST, 2)
            self.pantheon_founded = True
            self.env.city._event('pantheon_founded', belief=self.config.city_pantheon)
            self.env.on_pantheon_founded()

    def _update_loyalty(self):
        """Apply this turn's loyalty change; at 0 and still falling, the city declares independence."""
        change = self.loyalty_per_turn()
        self.loyalty = round(min(100, max(0, self.loyalty + change)), 2)
        if self.loyalty == 0 and change < 0 and not self.independent:
            self.independent = True
            self.env.city._event('city_declared_independence')

    # Reporting.
    def observation(self, build_items):
        """City-level features: size, progress towards growth, the item being built and the next tile,
        housing, mood, loyalty, war weariness, the governor and the pantheon, and which items are built."""
        item = self.current_build()
        progress = self.production / self.build_costs[item] if item else 0.0
        available, needed = self.amenities()
        pantheon = (1.0 if self.pantheon_founded else min(1.0, self.faith / PANTHEON_COST)
                    if self.config.city_pantheon else 0.0)
        return [self.population / 20, min(1.0, self.food / growth_threshold(self.population)),
                min(1.0, progress), min(1.0, self.culture / border_cost(self.tiles_acquired)),
                self.housing() / 20, *(float(item == name) for name in build_items),
                max(-1.0, min(1.0, (available - needed) / 10)), self.loyalty / 100,
                min(1.0, self.war_weariness / WAR_WEARINESS_PER_AMENITY / 10),
                float(self.governor_established()), pantheon,
                *(float(self.env.is_built(name)) for name in build_items)]

    def status_text(self):
        return '   '.join(text for _, text in self.status_fields())

    def status_fields(self):
        """The economy's status line as (icon name, text) fields."""
        yields = self.yields()
        item = self.current_build()
        building = (f'{item.replace("_", " ").title()} {self.production:.0f}/{self.build_costs[item]}'
                    if item else 'Nothing to build')
        available, needed = self.amenities()
        mood = self.mood()[0]
        fields = [('population', f'Pop {self.population}'), ('housing', f'Housing {self.housing():g}'),
                  ('food', f'Food {self.food:.0f}/{growth_threshold(self.population)} ({self.food_surplus(yields):+g})'),
                  ('production', f'{building} (+{yields.get("production", 0):g})'),
                  ('borders', f'Culture {self.culture:.0f}/{border_cost(self.tiles_acquired):.0f} '
                              f'(+{yields.get("culture", 0):g})'),
                  (f'mood_{mood}', f'Amenities {available}/{needed} {mood.title()}'),
                  ('loyalty', f'Loyalty {self.loyalty:g} ({self.loyalty_per_turn():+g})')]
        if self.war_weariness:
            fields.append(('war_weariness', f'War weariness {self.war_weariness}'))
        if yields.get('science'):
            fields.append(('science', f'Science +{yields["science"]:g}'))
        if self.config.city_pantheon:
            belief = self.config.city_pantheon.replace('_', ' ').capitalize()
            fields.append(('pantheon', belief) if self.pantheon_founded else
                          ('faith', f'Faith {self.faith:g}/{PANTHEON_COST} (+{yields.get("faith", 0):g}) toward {belief}'))
        elif yields.get('faith'):
            fields.append(('faith', f'Faith +{yields["faith"]:g}'))
        if self.config.city_governor:
            text = self.config.city_governor.capitalize()
            if self.config.city_governor_promotion:
                text += f' ({self.config.city_governor_promotion.replace("_", " ")})'
            if not self.governor_established():
                turns = GOVERNORS[self.config.city_governor].establish_turns - self.governor_turns
                text += f', establishing: {turns} turn{"" if turns == 1 else "s"}'
            fields.append(('governor', text))
        return fields

    def report(self):
        """The city's state, for summaries and the status panel."""
        yields = self.yields()
        available, needed = self.amenities()
        item = self.current_build()
        report = {'population': self.population, 'food': self.food, 'growth_threshold': growth_threshold(self.population),
                  'yields': yields, 'food_surplus': self.food_surplus(yields),
                  'growth_surplus': self.growth_surplus(yields), 'housing': self.housing(),
                  'growth_rate': growth_multiplier(self.population, self.housing()),
                  'amenities': available, 'amenities_needed': needed, 'mood': self.mood()[0],
                  'war_weariness': self.war_weariness, 'amenities_lost_to_war': self.amenities_lost_to_war(),
                  'loyalty': self.loyalty, 'loyalty_per_turn': self.loyalty_per_turn(),
                  'loyalty_sources': self.loyalty_sources(), 'territory': sorted(self.territory),
                  'tiles_acquired': self.tiles_acquired, 'culture': self.culture,
                  'border_cost': border_cost(self.tiles_acquired), 'border_growth': self.border_growth(yields),
                  'worked_tiles': self.worked_tiles(), 'blocked_tiles': self.blocked_tiles(),
                  'farms': sorted(self.farms, key=lambda p: (p[1], p[0])), 'districts': dict(self.districts),
                  'producing': item, 'production_progress': self.production,
                  'production_cost': self.build_costs[item] if item else None, 'completed': list(self.completed)}
        if self.config.city_governor:
            governor = self.config.city_governor
            report.update(governor=governor, governor_promotion=self.config.city_governor_promotion or None,
                          governor_established=self.governor_established(),
                          turns_to_establish=max(0, GOVERNORS[governor].establish_turns - self.governor_turns))
        if self.config.city_pantheon:
            report.update(faith_stored=self.faith, pantheon=self.config.city_pantheon,
                          pantheon_founded=self.pantheon_founded)
        return report
