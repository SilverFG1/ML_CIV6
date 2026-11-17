"""A rule-based empire policy: the rival's AI, the learner's demonstrations and its baseline.

It expands while there are good sites, improves tiles, builds a Monument and the
economic buildings its technologies allow, keeps one defender per city and
answers threats, then masses an army (with a battering ram against walls and
archers for support) and besieges the nearest rival city. Melee units only
assault a city when the retaliation is unlikely to kill them or a capture is
likely.
"""

import empire_rules as R
from class_hex import hex_distance
from empire import DECISION_KINDS, OPTION_INDEX, City

RESEARCH_ORDER = ('pottery', 'mining', 'animal_husbandry', 'bronze_working', 'archery', 'irrigation', 'writing',
                  'the_wheel', 'masonry', 'sailing', 'horseback_riding', 'iron_working', 'currency', 'shipbuilding',
                  'engineering')
# Spare military units needed before an attack starts, and the turn the build-up begins.
ATTACK_FORCE = 4
BUILDUP_TURN = 30
# A melee assault on a city needs this capture chance, or a death chance below the risk limit.
ASSAULT_CAPTURE = .5
ASSAULT_RISK = .15
MELEE_PREFERENCE = ('horseman', 'swordsman', 'heavy_chariot', 'spearman', 'warrior')


class Plan:
    """What the policy knows about one player's position this turn, shared by all its decisions."""

    def __init__(self, env, owner):
        self.env, self.owner = env, owner
        self.player = env.players[owner]
        self.cities = sorted(env.player_cities(owner), key=lambda c: c.id)
        land = [u for u in env.player_units(owner) if u.spec.military and u.spec.domain == 'land']
        self.threats = env.threats(owner)
        self.settlers = len(env.player_units(owner, 'settler')) + sum(c.item == 'settler' for c in self.cities)
        self.builders = len(env.player_units(owner, 'builder')) + sum(c.item == 'builder' for c in self.cities)
        self.military = len(land) + sum(c.item in R.LAND_MILITARY for c in self.cities)
        self.rams = sum(u.kind == 'battering_ram' for u in land) + sum(c.item == 'battering_ram' for c in self.cities)
        self.ranged = sum(R.UNITS[u.kind].ranged for u in land) + sum(
            c.item in R.UNITS and R.UNITS[c.item].ranged and R.UNITS[c.item].domain == 'land' for c in self.cities)
        # Each city keeps its nearest land unit as a guard; the rest are free to attack.
        self.guards = set()
        for city in self.cities:
            free = [u for u in land if u.id not in self.guards and u.kind != 'battering_ram'
                    and hex_distance(u.position, city.position) <= 3]
            if free:
                self.guards.add(min(free, key=lambda u: (hex_distance(u.position, city.position), u.id)).id)
        self.spare = [u for u in land if u.id not in self.guards]
        self.target = self._target()
        fighters = [u for u in self.spare if u.spec.can_attack]
        engaged = self.target is not None and any(hex_distance(u.position, self.target.position) <= 3
                                                  for u in fighters)
        self.offense = self.target is not None and (len(fighters) >= ATTACK_FORCE or (engaged and len(fighters) >= 2))
        self.buildup = self.target is not None and (env.turn_number >= BUILDUP_TURN or len(self.cities) >= 4)
        # Each new military unit costs a gold a turn; stop adding them when that would bankrupt us.
        self.affordable = env.net_income(owner) + self.player.gold / 20 >= 1

    def _target(self):
        """The rival city nearest our cities that our land units can reach."""
        env = self.env
        enemies = [c for c in env.cities if c.owner != self.owner]
        if not enemies or not self.cities:
            return None
        distances = env.enemy_city_distances(self.owner)
        reachable = [c for c in enemies if any(p in distances for p in env._neighbors(c.position))]
        home = [c.position for c in self.cities]
        reachable = [c for c in reachable if any(hex_distance(c.position, h) <= max(env.config.width,
                                                                                     env.config.height) for h in home)]
        if not reachable or env.separated(self.owner) and not env.can_embark(self.owner):
            return None
        return min(reachable, key=lambda c: (min(hex_distance(c.position, h) for h in home), c.id))

    def threats_near(self, position, radius=3):
        return [u for u in self.threats if hex_distance(u.position, position) <= radius]

    def defenders_near(self, position):
        return [u for u in self.env.player_units(self.owner) if u.spec.military and u.spec.domain == 'land'
                and hex_distance(u.position, position) <= 1]


def choose(env, owner, decisions, rng):
    """One option index per decision."""
    plan = Plan(env, owner)
    return tuple(_choose(env, plan, decision, rng) for decision in decisions)


def _option(decision, name):
    return OPTION_INDEX[decision.kind][name]


def _names(decision):
    options = DECISION_KINDS[decision.kind].options
    return [options[index] for index in decision.valid]


def _choose(env, plan, decision, rng):
    if len(decision.valid) == 1:
        return decision.valid[0]
    handler = {'research': _research, 'city': _city, 'settler': _settler, 'builder': _builder,
               'land': _land, 'naval': _naval, 'trader': _trader}[decision.kind]
    name = handler(env, plan, decision, rng)
    index = _option(decision, name)
    return index if index in decision.valid else decision.valid[-1]


def _research(env, plan, decision, rng):
    resources = {env.resources[p] for c in plan.cities for p in c.territory if p in env.resources
                 and p not in env.improvements}
    wanted = {R.IMPROVEMENTS[R.RESOURCES[name].improvement].technology for name in resources}
    separated = env.separated(plan.owner)
    coastal = any(env.coastal(c.position) for c in plan.cities)

    def rank(tech):
        score = RESEARCH_ORDER.index(tech)
        if tech in wanted:
            score -= 5
        if separated and tech in ('sailing', 'shipbuilding'):
            score -= 8
        if plan.threats and tech in ('bronze_working', 'archery'):
            score -= 4
        if coastal and tech == 'sailing':
            score -= 2
        return score + rng.random() * .01
    return min(_names(decision), key=rank)


def _best_melee(names):
    return next((name for name in MELEE_PREFERENCE if name in names), 'warrior')


def _city(env, plan, decision, rng):
    city = next(c for c in plan.cities if c.id == decision.entity)
    names = _names(decision)
    threat = len(plan.threats_near(city.position))
    defenders = len(plan.defenders_near(city.position))
    if city.item is not None:
        price = env.item_price(plan.owner, city.item)
        gold = plan.player.gold
        urgent = city.item in R.LAND_MILITARY and threat > defenders
        if 'buy' in names and (urgent or (city.item == 'settler' and gold >= price + 30)
                               or (city.item == 'builder' and gold >= price + 60)):
            return 'buy'
        return city.item
    choice = _city_item(env, plan, city, names, threat, defenders)
    if choice in R.LAND_MILITARY:
        plan.military += 1
        plan.rams += choice == 'battering_ram'
        plan.ranged += R.UNITS[choice].ranged
    plan.settlers += choice == 'settler'
    plan.builders += choice == 'builder'
    return choice


def _city_item(env, plan, city, names, threat, defenders):
    if threat > defenders:
        if 'ancient_walls' in names and threat >= 2 and city.population >= 3:
            return 'ancient_walls'
        if 'archer' in names:
            return 'archer'
        return _best_melee(names)
    if defenders == 0 and plan.military < len(plan.cities):
        return _best_melee(names)
    sites = sum(hex_distance(p, city.position) <= 6 and env.legal_site(plan.owner, p) for p in env.site_values())
    wanted_settlers = 1 + (len(plan.cities) >= 3)
    if 'settler' in names and sites and plan.settlers < wanted_settlers and env.turn_number < env.config.turn_limit - 15:
        return 'settler'
    improvable = sum(env.improvable(plan.owner, p) is not None for p in city.territory)
    if 'builder' in names and improvable >= 2 and plan.builders < len(plan.cities) // 2 + 1:
        return 'builder'
    if 'monument' in names:
        return 'monument'
    if plan.buildup and plan.affordable and plan.military < len(plan.cities) + ATTACK_FORCE + 1:
        if plan.target.wall_hp_max > 0 and plan.rams == 0 and 'battering_ram' in names:
            return 'battering_ram'
        if plan.ranged < max(1, plan.military // 3) and 'archer' in names:
            return 'archer'
        return _best_melee(names)
    if 'granary' in names and env.housing(city) - city.population <= 1:
        return 'granary'
    for name in ('water_mill', 'library', 'trader', 'market'):
        if name in names:
            return name
    if 'ancient_walls' in names and (city.has_palace or threat):
        return 'ancient_walls'
    if 'settler' in names and sites:
        return 'settler'
    if 'galley' in names and env.separated(plan.owner) and not any(
            u.kind in R.NAVAL_UNITS for u in env.player_units(plan.owner)):
        return 'galley'
    if 'granary' in names:
        return 'granary'
    if plan.affordable and plan.military < 2 * len(plan.cities) + 2:
        return _best_melee(names)
    if 'builder' in names and improvable:
        return 'builder'
    return 'ancient_walls' if 'ancient_walls' in names else _best_melee(names)


def _settler(env, plan, decision, rng):
    unit = next(u for u in env.units if u.id == decision.entity)
    names = _names(decision)
    ranked = env.ranked_sites(unit)
    if 'found' in names:
        here = env.site_values().get(unit.position, 0)
        if not ranked or ranked[0][1] == unit.position or here >= .9 * ranked[0][2]:
            return 'found'
    if 'site_1' in names:
        return 'site_1'
    return 'found' if 'found' in names else 'hold'


def _builder(env, plan, decision, rng):
    names = _names(decision)
    for name in ('improve', 'goto_resource', 'goto_tile'):
        if name in names:
            return name
    return 'hold'


def _trader(env, plan, decision, rng):
    return 'route' if 'route' in _names(decision) else 'hold'


def _land(env, plan, decision, rng):
    unit = next(u for u in env.units if u.id == decision.entity)
    names = _names(decision)
    in_city = env.city_at.get(unit.position)
    home = in_city is not None and in_city.owner == plan.owner
    best = env.best_attack(unit)
    near_threats = [t for t in plan.threats if hex_distance(t.position, unit.position) <= 4]
    if near_threats and 'defend' in names and unit.hp >= 50:
        if best is not None and not isinstance(best[0], City) and (unit.spec.ranged or best[2] < ASSAULT_RISK
                                                                    or best[1] >= ASSAULT_CAPTURE):
            return 'defend'
        if unit.spec.ranged or unit.id in plan.guards:
            return 'garrison' if not home else ('defend' if unit.spec.ranged else 'fortify')
    if unit.hp < 40:
        holder = env.tile_owner.get(unit.position)
        return 'fortify' if home or (holder is not None and holder.owner == plan.owner) else 'garrison'
    if unit.id in plan.guards:
        escort = _escort_needed(env, plan, unit)
        if escort and 'escort' in names:
            return 'escort'
        return 'garrison' if not home else 'fortify'
    if plan.offense and 'siege' in names:
        if unit.hp < 60:
            return 'fortify'
        if best is not None and isinstance(best[0], City) and not unit.spec.ranged:
            _, capture, death = best
            if capture >= ASSAULT_CAPTURE or death < ASSAULT_RISK:
                return 'siege'
            return 'fortify'
        return 'siege'
    if best is not None and not isinstance(best[0], City) and 'hunt' in names and (
            unit.spec.ranged or best[1] >= ASSAULT_CAPTURE or best[2] < ASSAULT_RISK):
        return 'hunt'
    if _escort_needed(env, plan, unit) and 'escort' in names:
        return 'escort'
    return 'garrison' if 'garrison' in names and not home else 'fortify'


def _escort_needed(env, plan, unit):
    """Whether unit is the nearest military unit to a settler that has none beside it."""
    for settler in env.player_units(plan.owner, 'settler'):
        guarded = any(other.owner == plan.owner and other is not unit for p in [settler.position]
                      + env._neighbors(settler.position) for other in (env.military_at.get(p),) if other is not None)
        if guarded or hex_distance(settler.position, unit.position) > 4:
            continue
        land = [u for u in env.player_units(plan.owner) if u.spec.military and u.spec.domain == 'land']
        nearest = min(land, key=lambda u: (hex_distance(u.position, settler.position), u.id))
        if nearest is unit:
            return True
    return False


def _naval(env, plan, decision, rng):
    unit = next(u for u in env.units if u.id == decision.entity)
    names = _names(decision)
    if unit.hp < 50 and 'patrol' in names:
        return 'patrol'
    best = env.best_attack(unit)
    if best is not None and 'hunt' in names and (unit.spec.ranged or best[2] < ASSAULT_RISK or best[1] >= .5):
        return 'hunt' if not isinstance(best[0], City) else ('raid' if 'raid' in names else 'hunt')
    if plan.offense and 'raid' in names:
        return 'raid'
    if plan.threats and 'hunt' in names:
        return 'hunt'
    return 'patrol' if 'patrol' in names else 'fortify'
