"""Rules data for the empire simulation: units, buildings, improvements, resources and technologies.

Values follow Civilization VI's base game (standard speed) where noted in the
README; the remaining values are this simulator's own choices, and the README
lists them as such. Nothing here imports graphics or the siege environment.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UnitType:
    cost: int
    strength: float = 0
    ranged_strength: float = 0
    attack_range: int = 0
    moves: int = 2
    domain: str = 'land'           # land or sea
    unit_class: str = 'melee'      # melee, anti_cavalry, light_cavalry, heavy_cavalry, ranged, support,
    #                                civilian, naval_melee or naval_ranged
    technology: str | None = None
    maintenance: int = 1           # gold per turn
    charges: int = 0               # builder charges
    resource: str | None = None    # strategic resource that must be improved in the owner's territory

    @property
    def military(self):
        return self.unit_class != 'civilian'

    @property
    def ranged(self):
        return self.ranged_strength > 0

    @property
    def can_attack(self):
        return self.military and self.unit_class != 'support'


UNITS = {
    'settler': UnitType(80, moves=2, unit_class='civilian', maintenance=0),
    'builder': UnitType(50, moves=2, unit_class='civilian', maintenance=0, charges=3),
    'trader': UnitType(40, moves=2, unit_class='civilian', maintenance=0, technology='currency'),
    'warrior': UnitType(40, 20),
    'slinger': UnitType(35, 5, ranged_strength=15, attack_range=1, unit_class='ranged'),
    'archer': UnitType(60, 15, ranged_strength=25, attack_range=2, unit_class='ranged', technology='archery'),
    'spearman': UnitType(65, 25, unit_class='anti_cavalry', technology='bronze_working'),
    'heavy_chariot': UnitType(65, 28, unit_class='heavy_cavalry', technology='the_wheel'),
    'horseman': UnitType(80, 36, moves=4, unit_class='light_cavalry', technology='horseback_riding',
                         resource='horses'),
    'swordsman': UnitType(90, 35, unit_class='melee', technology='iron_working', resource='iron'),
    'battering_ram': UnitType(65, 0, unit_class='support', technology='masonry'),
    'galley': UnitType(65, 30, moves=3, domain='sea', unit_class='naval_melee', technology='sailing'),
    'quadrireme': UnitType(120, 20, ranged_strength=25, attack_range=1, moves=3, domain='sea',
                           unit_class='naval_ranged', technology='shipbuilding'),
}
UNIT_NAMES = tuple(UNITS)
LAND_MILITARY = tuple(name for name, spec in UNITS.items() if spec.military and spec.domain == 'land')
NAVAL_UNITS = tuple(name for name, spec in UNITS.items() if spec.domain == 'sea')
# Settlers cost 30 more for every settler the civilization has already trained.
SETTLER_COST_STEP = 30
# A settler cannot be started in a city of one citizen; finishing one costs a citizen.
SETTLER_MIN_POPULATION = 2
# Spearmen and other anti-cavalry units gain this much strength against cavalry.
ANTI_CAVALRY_BONUS = 10
CAVALRY_CLASSES = ('light_cavalry', 'heavy_cavalry')
# Embarked land units defend with this strength and cannot attack.
EMBARKED_STRENGTH = 10
# Units lose one strength for every this many hit points of damage.
WOUND_STEP = 10


@dataclass(frozen=True)
class Building:
    cost: int
    yields: tuple = ()             # (yield, amount) pairs
    housing: float = 0
    technology: str | None = None
    maintenance: int = 0
    requires: str | None = None    # 'river' or 'coast' next to the city centre


BUILDINGS = {
    'monument': Building(60, (('culture', 2),)),
    'granary': Building(65, (('food', 1),), housing=2, technology='pottery', maintenance=1),
    'water_mill': Building(80, (('food', 1), ('production', 1)), technology='the_wheel', maintenance=1,
                           requires='river'),
    # Civ VI builds these in Campus and Commercial Hub districts, which this simulator does not have.
    'library': Building(90, (('science', 2),), technology='writing', maintenance=1),
    'market': Building(120, (('gold', 3),), technology='currency'),
    'ancient_walls': Building(80, technology='masonry'),
}
BUILDING_NAMES = tuple(BUILDINGS)
# The capital's Palace; it cannot be built.
PALACE_YIELDS = (('production', 2), ('science', 2), ('gold', 5), ('culture', 1))
WALL_HP = 100
# Walls strike with the owner's strongest available ranged unit, or 15 without one.
WALL_MIN_RANGED_STRENGTH = 15
CITY_RANGED_RANGE = 2
CITY_HP = 100
# A city defends with the strongest land melee unit its owner can train, plus this.
CITY_STRENGTH_BONUS = 8
CITY_HEAL = 10
# Gold buys an item for this many gold per point of its production cost.
GOLD_PER_PRODUCTION = 4


@dataclass(frozen=True)
class Improvement:
    technology: str | None
    yields: tuple
    housing: float = 0


IMPROVEMENTS = {
    'farm': Improvement(None, (('food', 1),), housing=.5),
    'mine': Improvement('mining', (('production', 1),)),
    'quarry': Improvement('mining', (('production', 1),)),
    'pasture': Improvement('animal_husbandry', (('production', 1),), housing=.5),
    'plantation': Improvement('irrigation', (('gold', 2),), housing=.5),
    'fishing_boats': Improvement('sailing', (('food', 1),), housing=.5),
}
IMPROVEMENT_NAMES = tuple(IMPROVEMENTS)
# Land types that take a farm without a resource, flat or hills.
FARM_LAND = ('GRASSLAND', 'PLAINS')


@dataclass(frozen=True)
class Resource:
    kind: str                      # bonus, strategic or luxury
    yields: tuple
    improvement: str
    terrains: tuple                # terrain names it can appear on
    feature: str | None = None     # a feature it needs, such as FOREST


_FLAT_FARMLAND = ('GRASSLAND', 'PLAINS')
_HILLS = ('GRASSLAND_HILLS', 'PLAINS_HILLS', 'DESERT_HILLS', 'TUNDRA_HILLS')
RESOURCES = {
    'wheat': Resource('bonus', (('food', 1),), 'farm', ('PLAINS', 'GRASSLAND')),
    'rice': Resource('bonus', (('food', 1),), 'farm', ('GRASSLAND',)),
    'cattle': Resource('bonus', (('food', 1),), 'pasture', ('GRASSLAND',)),
    'sheep': Resource('bonus', (('food', 1),), 'pasture', _HILLS),
    'stone': Resource('bonus', (('production', 1),), 'quarry', ('GRASSLAND', 'PLAINS', 'DESERT') + _HILLS),
    'fish': Resource('bonus', (('food', 1),), 'fishing_boats', ('COAST',)),
    'crabs': Resource('bonus', (('gold', 2),), 'fishing_boats', ('COAST',)),
    'horses': Resource('strategic', (('food', 1), ('production', 1)), 'pasture', _FLAT_FARMLAND),
    'iron': Resource('strategic', (('science', 1),), 'mine', _HILLS),
    'citrus': Resource('luxury', (('food', 2),), 'plantation', _FLAT_FARMLAND),
    'cotton': Resource('luxury', (('gold', 3),), 'plantation', _FLAT_FARMLAND),
    'silk': Resource('luxury', (('culture', 1),), 'plantation', ('GRASSLAND', 'PLAINS'), feature='FOREST'),
    'salt': Resource('luxury', (('food', 1), ('gold', 1)), 'mine', ('PLAINS', 'DESERT', 'TUNDRA')),
    'pearls': Resource('luxury', (('faith', 1),), 'fishing_boats', ('COAST',)),
}
RESOURCE_NAMES = tuple(RESOURCES)
LUXURY_NAMES = tuple(name for name, spec in RESOURCES.items() if spec.kind == 'luxury')
# Each improved luxury type gives one amenity to up to this many of its owner's cities.
LUXURY_CITIES = 4


@dataclass(frozen=True)
class Technology:
    cost: int
    prerequisites: tuple = ()
    boost: str | None = None       # event that pays half the cost (a Eureka)


TECHNOLOGIES = {
    'pottery': Technology(25),
    'animal_husbandry': Technology(25),
    'mining': Technology(25),
    'sailing': Technology(50, boost='coastal_city'),
    'irrigation': Technology(50, ('pottery',), boost='farm_resource'),
    'writing': Technology(50, ('pottery',), boost='met_civilization'),
    'archery': Technology(50, ('animal_husbandry',), boost='slinger_kill'),
    'masonry': Technology(80, ('mining',), boost='quarry'),
    'bronze_working': Technology(80, ('mining',)),
    'the_wheel': Technology(80, ('mining',), boost='mine_resource'),
    'currency': Technology(120, ('writing',), boost='trade_route'),
    'horseback_riding': Technology(120, ('animal_husbandry',), boost='pasture'),
    'iron_working': Technology(120, ('bronze_working',), boost='mine_iron'),
    'shipbuilding': Technology(200, ('sailing',), boost='two_galleys'),
    'engineering': Technology(200, ('the_wheel',), boost='ancient_walls'),
}
TECHNOLOGY_NAMES = tuple(TECHNOLOGIES)
BOOST_SHARE = .5
# Land units embark onto coast with this technology. No unit enters ocean (that needs Cartography).
EMBARK_TECHNOLOGY = 'shipbuilding'
# Roads bridge rivers once this technology is known.
BRIDGE_TECHNOLOGY = 'engineering'

# Each citizen produces this much science.
SCIENCE_PER_CITIZEN = .5
# A domestic trade route adds these yields to its origin city each turn.
TRADE_ROUTE_YIELDS = (('food', 1), ('production', 1))
# Trade-route capacity: one, plus one per Market.
BASE_TRADE_ROUTES = 1

# Turn-start healing for a unit that neither moved nor attacked, by location.
HEAL_IN_CITY, HEAL_FRIENDLY, HEAL_NEUTRAL, HEAL_ENEMY = 20, 15, 10, 5

# Amenity mood: (lowest amenity surplus, mood, growth multiplier, non-food yield multiplier).
AMENITY_MOODS = (
    (3, 'ecstatic', 1.2, 1.1),
    (1, 'happy', 1.1, 1.05),
    (0, 'content', 1.0, 1.0),
    (-2, 'displeased', .85, .95),
    (-4, 'unhappy', .7, .9),
    (-6, 'unrest', 0.0, .85),
    (-math.inf, 'revolt', 0.0, .8),
)

SCORE_WEIGHTS = {'city': 5, 'citizen': 1, 'technology': 2, 'building': 1}
# Cities must be at least this far apart.
MIN_CITY_DISTANCE = 4


def amenities_required(population):
    """One amenity for every two citizens, the first at three citizens."""
    return max(0, (population - 1) // 2)


def amenity_mood(surplus):
    """Return (mood, growth multiplier, non-food yield multiplier) for an amenity surplus."""
    for lowest, mood, growth, yields in AMENITY_MOODS:
        if surplus >= lowest:
            return mood, growth, yields
    raise AssertionError('unreachable')


def available_technologies(known):
    """Technologies not yet known whose prerequisites are all known, in table order."""
    return tuple(name for name, tech in TECHNOLOGIES.items()
                 if name not in known and all(p in known for p in tech.prerequisites))


def unit_cost(name, settlers_trained=0):
    cost = UNITS[name].cost
    return cost + SETTLER_COST_STEP * settlers_trained if name == 'settler' else cost


def item_cost(name, settlers_trained=0):
    return unit_cost(name, settlers_trained) if name in UNITS else BUILDINGS[name].cost


def add_yields(totals, yields):
    for kind, amount in yields:
        totals[kind] = totals.get(kind, 0) + amount
    return totals
