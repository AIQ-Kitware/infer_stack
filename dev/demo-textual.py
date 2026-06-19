from __future__ import annotations

"""
Single-file RPG architecture prototype.

This version integrates the TUI with the game a little bit better and has some
visual art.

Reading guide
-------------
v7 is v6 with one extra: an ASCII_ART_POLISHED dictionary (search the
file) that maps event keys -> multi-line ASCII art strings. The engines
look these up to render an illustration alongside text for some
encounters and NPCs. Everything else is the v4 RPG core + v5 Textual
front-end fused together — see v3, v4, v5 for the architecture comments.
"""

from collections import Counter
from dataclasses import dataclass, field
import random
import time
from typing import Callable


# ============================================================
# Small helpers / presentation
# ============================================================


def typewriter_print(prefix: str, text: str, word_delay: float = 0.018) -> None:
    print(prefix, end="")
    for word in text.split():
        print(word, end=" ", flush=True)
        time.sleep(word_delay)
    print()


def say(name: str, text: str) -> None:
    typewriter_print(f"{name}: ", text)


def divider(title: str | None = None) -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("-" * 72)


def action_separator(label: str | None = None) -> None:
    print("\n" + "-" * 72)
    if label:
        print(f"> {label}")
        print("-" * 72)


def prompt_continue(prompt: str = "Press Enter to continue...") -> None:
    input(f"\n{prompt}")


# ============================================================
# Visual art (ASCII)
# ============================================================
# Maps short string keys ("ruins_stumble", an NPC name, etc.) to
# multi-line ASCII art. r"""..."""  is a *raw* triple-quoted string —
# the `r` prefix means backslashes are kept literal, which matters here
# because the art uses backslashes as drawing characters.
# Engines below look these keys up to render art when an NPC is
# inspected or an encounter fires.
ASCII_ART_POLISHED = {
    "ruins_stumble": r"""\
      __
   __/ /__
 _/  _   _\_
/__ /_\ /_\_\
   \/ / \/
    /_/\
   /_/ /
""",
    "bramble_push": r"""\
 \\\   |   ////
  \\\  |  ////
---\\ | ////---
----\\|////----
----////|\\----
---//// | \\---
  ////  |  \\
""",
    "dark_cave_blocked": r"""\
    _________
   /  _____  \
  /  /     \  \
 |  |  XXX  |  |
 |  |  XXX  |  |
 |  |_______|  |
  \___________/
""",
    "broken_crossing": r"""\
  __         __
_/ /________/ /_
\_\   ____   /_/
  /  / __ \  \
_/__/ /  \ \__\_
\____/    \____/
""",
    "tower_gate_unlock": r"""\
      /\
     /[]\
    /_[]_\
    | || |
    | || |
    | || |
    |_/\_|
""",
    "forest_whisper_hint": r"""\
    ^^     ^^
   <  \   /  >
      { o }
     /  |  \
    /   |   \
       / \
""",
    "trail_reopened": r"""\
    _________
 __/  __ __  \__
/_   / // /   _\
  \ /_//_/ \ /
   /  __    \
 _/__/  \____\_
\______________/
""",
    "serve_time": r"""\
  | | | | | |
  | | | | | |
  | | |o| | |
  | | /|\ | |
  | | / \ | |
  |_|_____|_|
""",
    "jail_release": r"""\
  | | |   | |
  | | |   | |
  | | |o  | |
  | | /|\   |
  | | / \   |
  |_|_______|
""",
    "first_tower_visit": r"""\
       /\
      /  \
     / ** \
    / **** \
   /______ \
      ||||
    ~~~~~~~~
""",
    "bandit_surrenders": r"""\
    .------.
   / @   @ \
   |   ^   |
   |  ---  |
   \_/| |\_/
      | |
     /   \
""",
    "player_surrender_accepted": r"""\
      \o/
       |
      / \
     /___\
       |
      / \
""",
    "player_surrender_rejected": r"""\
      \o/
       |
      / \
    X/   \X
    /_____\
      / \
""",
    "guardian_trial_passed": r"""\
     .-***-.
   .'  ***  '.
  /   ** **   \
 <     ***     >
  \   ** **   /
   '.  ***  .'
     '-***-'
""",
    "guardian_trial_failed": r"""\
     .--?--.
   .'   !   '.
  /    /_\    \
 <      !      >
  \    \_/    /
   '.   !   .'
     '--.--'
""",
    "spider_web_cleared": r"""\
 \   |  |  |   /
  \  |  |  |  /
---\ \  ()  / /---
---/ / /\/\ \ \---
  /  |  |  |  \
 /   |  |  |   \
""",
    "return_crystal": r"""\
       /\
      /**\
     /****\
      \__/
       ||
   .-========-.
  /  ________  \
  \_/________\_/
""",
    "fountain_restored": r"""\
       /\
      /**\
     /****\
    ~~~||~~~
  ~~~~~||~~~~~
      _||_
    _/____\_
""",
    "fountain_rest": r"""\
     _=====_
   _/ _____ \_
  /  /     \  \
 |  |  ~~~  |  |
 |  |  ~~~  |  |
  \  \_____/  /
   \_________/
""",
    "quest_complete_cheer": r"""\
   \o/   \o/
    |     |
   / \   / \
  * * * * * *
   \o/   \o/
    |     |
   / \   / \
""",
    "guard_arrest": r"""\
     [###]
      /|\
  ___  |   ___
 [___]/ \ [___]
   |         |
  / \       / \
""",
    "item_shatter": r"""\
      __
     |++|
   __|__|__
  /  /\/\  \
  \_/\__/\\_/
     /  \
""",
    "mercy_choice": r"""\
      \o
       |\
      / \
    __/ \__
   /  mercy \
   \________/
""",
    # Dedicated location icons
    "loc::village": r"""\
      _/\_   _/\_
     /    \ /    \
    | []  | | [] |
    |_____| |____|
       \  ___  /
        /~ ~ \
       /_____\
""",
    "loc::crossroads": r"""\
         ||
    =====++=====
         ||
      ---++---
         ||
        /  \
""",
    "loc::forest": r"""\
       &&&  &&&
     &&&&&&&&&&&
       || || ||
      /| || ||\
     /_|_||_||_\
""",
    "loc::garden": r"""\
       .-*-.
    .-(  *  )-.
   / *  /\  * \
   \___/  \___/
      /_/\_\
     <*>  <*>
""",
    "loc::lake": r"""\
       _      _
    .-(.)----(.)-.
 ~~~/            \~~~
   /   ~~~~~~~~   \
  /________________\
      /_/    \_\
""",
    "loc::cave_entrance": r"""\
       ________
     /  ____   \
    /  /    \   \
   |  |      |  |
   |  |  __  |  |
    \  \____/  /
     \________/
""",
    "loc::cave_depths": r"""\
        /\/\
       /_**_\
      /_****_\
      \ *  * /
       \_**_/
      /_/  \_\
""",
    "loc::ruins": r"""\
       _[]_
      |    |
    __|_  _|__
   / _  ||  _ \
   ||_|_||_|_||
    \__      __/
       |____|
""",
    "loc::tower_gate": r"""\
        /^^\
       /_[]_\
       | || |
       | || |
       | || |
       |_||_|
       /_/\_\
""",
    "loc::tower_top": r"""\
         /\
        /  \
       /_[]_\
         ||
       __||__
      /______\
        /  \
""",
    "loc::jail": r"""\
      .------.
      | |||| |
      | |||| |
      | |||| |
      | |||| |
      |______|
""",
    # NPC-specific placeholder art
    "npc::elder_mira::alive": r"""\
      .-====-.
    .'  .--.  '.
   /   /    \   \
  |   | o  o |   |
  |   |  --  |   |
  |   | \__/ |   |
   \   \____/   /
    '.  ____  .'
      '------'
""",
    "npc::elder_mira::dead": r"""\
      .-====-.
    .'  .--.  '.
   /   /    \   \
  |   | x  x |   |
  |   |  --  |   |
  |   | \__/ |   |
   \   \____/   /
    '.  ____  .'
      '------'
""",
    "npc::guard_halwen::alive": r"""\
      .------.
     /  ____  \
    |  |o  o|  |
    |  | -- |  |
    |  |____|  |
    | /|_||_|\ |
     \_  ||  _/
       | || |
       |_||_|
""",
    "npc::guard_halwen::dead": r"""\
      .------.
     /  ____  \
    |  |x  x|  |
    |  | -- |  |
    |  |____|  |
    | /|_.._|\ |
     \_  ||  _/
       |    |
       |_/\_|
""",
    "npc::merchant_sella::alive": r"""\
      .------.
     /  o  o \\
    |    ..   |
    |  \____/ |
    | .-====-.|
     \| $$$$ |/
      '------'
       /||||\\
""",
    "npc::merchant_sella::dead": r"""\
      .------.
     /  x  x \\
    |    ..   |
    |  \____/ |
    | .-====-.|
     \|_$$$$_|/
      '------'
       /    \\
""",
    "npc::fisher_rowan::alive": r"""\
      .------.
     /  o  o \\
    |    ><   |
    |  \____/ |
     \   ||   /
      '._||_.'
         ||~~~
         ||
""",
    "npc::fisher_rowan::dead": r"""\
      .------.
     /  x  x \\
    |    ><   |
    |  \____/ |
     \   ||   /
      '._||_.'
         ||
         ~~
""",
    "npc::bandit_nox::alive": r"""\
    .------.
   / o   o \
   |   ^   |
   |  ---  |
   \_/| |\_/
      | |
     /   \

""",
    "npc::bandit_nox::dead": r"""\
    .------.
   / x   x \
   |   ^   |
   |  ---  |
   \_/| |\_/
      | |
     /   \

""",
    "npc::crystal_spider::alive": r"""\
   \\  |\  /|  //
    \\ | \/ | //
-----\\| () |//-----
-----//|/==\\|\\-----
    // |/  \\| \\
   //  /_/\\_\\  \\
""",
    "npc::crystal_spider::dead": r"""\
   \\  |\  /|  //
    \\ | xx | //
-----\\|_==_|//-----
-----//|/--\\|\\-----
    // |/  \\| \\
   //  /_/\\_\\  \\
""",
    "npc::tower_guardian::alive": r"""\
       .-**-.
     .' *  * '.
    /  * /\\ *  \\
   <   * || *   >
    \\  * \/ *  //
     '. *  * .'
       '-**-'
""",
    "npc::tower_guardian::dead": r"""\
       .-..-.
     .' xx xx'.
    /  .. /\\ ..\\
   <   .. || ..  >
    \\  .. \/ ..//
     '. xx xx .'
       '-..-'
""",
    "npc::small_spider::alive": r"""\
   \\  /\  //
----\\(oo)//----
----// || \\\----
   // _||_ \\
""",
    "npc::small_spider::dead": r"""\
   \\  /\  //
----\\(xx)//----
----//_||_\\----
   // /  \\
""",
}


# ============================================================
# Core data models
# ============================================================


@dataclass
class Stats:
    max_hp: int
    attack_min: int
    attack_max: int
    defense: int = 0


@dataclass
class Item:
    item_id: str
    name: str
    description: str = ""
    slot: str | None = None
    power_bonus: int = 0
    defense_bonus: int = 0
    hp_bonus: int = 0
    healing: int = 0
    set_flags_on_pickup: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)


@dataclass
class TradeOffer:
    title: str
    wants_items: list[str] = field(default_factory=list)
    gives_items: list[str] = field(default_factory=list)
    wants_gold: int = 0
    gives_gold: int = 0
    repeatable: bool = False
    required_flags: set[str] = field(default_factory=set)
    blocked_flags: set[str] = field(default_factory=set)


@dataclass
class DialogueTopic:
    key: str
    title: str
    lines: list[str]
    required_flags: set[str] = field(default_factory=set)
    blocked_flags: set[str] = field(default_factory=set)
    required_items: list[str] = field(default_factory=list)
    once: bool = False
    outcome_effect: Effect | None = None


@dataclass
class Riddle:
    question: str
    answers: list[str]
    intro_lines: list[str] = field(default_factory=list)
    success_lines: list[str] = field(default_factory=list)
    failure_lines: list[str] = field(default_factory=list)
    repeat_lines: list[str] = field(default_factory=list)
    damage_on_failure: int = 0
    set_flags_on_success: set[str] = field(default_factory=set)


@dataclass
class Exit:
    direction: str
    destination: str
    requires_item: str | None = None
    requires_flag: str | None = None
    blocked: bool = False
    blocked_text: str = "That path is blocked."
    warning_text: str | None = None
    on_attempt_effect: Effect | None = None
    on_blocked_effect: Effect | None = None
    on_success_effect: Effect | None = None


@dataclass
class Location:
    key: str
    name: str
    description: str
    exits: list[Exit] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    npcs: list[NPC] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)


# ============================================================
# Effects
# ============================================================


class Effect:
    def apply(self, game: Game) -> None:
        raise NotImplementedError


class PrintEffect(Effect):
    def __init__(self, lines: str | list[str]):
        self.lines = [lines] if isinstance(lines, str) else lines

    def apply(self, game: Game) -> None:
        for line in self.lines:
            print(line)


class SetFlagEffect(Effect):
    def __init__(self, *flags: str):
        self.flags = list(flags)

    def apply(self, game: Game) -> None:
        for flag in self.flags:
            game.flags.add(flag)


class ClearFlagEffect(Effect):
    def __init__(self, *flags: str):
        self.flags = list(flags)

    def apply(self, game: Game) -> None:
        for flag in self.flags:
            game.flags.discard(flag)


class SetBountyEffect(Effect):
    def __init__(self, amount: int):
        self.amount = amount

    def apply(self, game: Game) -> None:
        game.bounty = max(0, self.amount)
        print(f"[Bounty] Set to {game.bounty}")


class ChangeBountyEffect(Effect):
    def __init__(self, amount: int, reason: str | None = None):
        self.amount = amount
        self.reason = reason

    def apply(self, game: Game) -> None:
        game.change_bounty(self.amount, self.reason)


class HealPlayerEffect(Effect):
    def __init__(
        self, amount: int, heal_text: str | None = None, full_text: str | None = None
    ):
        self.amount = amount
        self.heal_text = heal_text or "You recover some health."
        self.full_text = full_text or "You already feel fully rested."

    def apply(self, game: Game) -> None:
        if game.player.health >= game.player.total_max_hp(game.world.item_db):
            print(self.full_text)
            return
        print(self.heal_text)
        healed = game.player.heal(self.amount, game.world.item_db)
        print(f"Recovered {healed} health.")


class GivePlayerItemEffect(Effect):
    def __init__(self, *item_ids: str):
        self.item_ids = list(item_ids)

    def apply(self, game: Game) -> None:
        for item_id in self.item_ids:
            game.give_player_item(item_id)


class ChangeGoldEffect(Effect):
    def __init__(
        self, amount: int, *, target: str = "player", reason: str | None = None
    ):
        self.amount = amount
        self.target = target
        self.reason = reason

    def apply(self, game: Game) -> None:
        actor = game.player if self.target == "player" else game.find_npc(self.target)
        if actor is None:
            return
        actor.gold = max(0, actor.gold + self.amount)
        label = f"[Gold] {actor.name} now has {actor.gold} gold"
        if self.reason:
            print(f"{self.reason}")
            print(label)
        elif self.target == "player":
            if self.amount > 0:
                print(f"You gain {self.amount} gold.")
            elif self.amount < 0:
                print(f"You lose {-self.amount} gold.")
            print(label)


class RemoveItemEffect(Effect):
    def __init__(
        self, item_id: str, *, target: str = "player", text: str | None = None
    ):
        self.item_id = item_id
        self.target = target
        self.text = text

    def apply(self, game: Game) -> None:
        actor = game.player if self.target == "player" else game.find_npc(self.target)
        if actor is None:
            return
        if actor.remove_item(self.item_id) and self.text:
            print(self.text)


class RemoveRandomItemEffect(Effect):
    def __init__(self, *, target: str = "player", text_prefix: str = "You lose"):
        self.target = target
        self.text_prefix = text_prefix

    def apply(self, game: Game) -> None:
        actor = game.player if self.target == "player" else game.find_npc(self.target)
        if actor is None or not actor.inventory:
            return
        item_id = random.choice(actor.inventory)
        actor.remove_item(item_id)
        print(f"{self.text_prefix}: {game.item_name(item_id)}")


class DamageCharacterEffect(Effect):
    def __init__(self, target: str, amount: int, text: str | None = None):
        self.target = target
        self.amount = amount
        self.text = text

    def apply(self, game: Game) -> None:
        actor: Character | None
        if self.target == "player":
            actor = game.player
        else:
            actor = game.find_npc(self.target)
        if actor is None:
            return
        if self.text:
            print(self.text)
        actual = actor.take_damage(self.amount, game.world.item_db)
        print(f"{actor.name} takes {actual} damage.")
        if isinstance(actor, NPC) and not actor.is_alive() and not actor.defeated:
            actor.on_defeat(game)


class MovePlayerEffect(Effect):
    def __init__(self, destination: str, text: str | None = None):
        self.destination = destination
        self.text = text

    def apply(self, game: Game) -> None:
        if self.text:
            print(self.text)
        game.player.location = self.destination
        game.flags.add(f"visited:{self.destination}")
        game.world.on_location_enter(game, game.current_location())


class BlockPathEffect(Effect):
    def __init__(self, location_key: str, direction: str, reason: str):
        self.location_key = location_key
        self.direction = direction
        self.reason = reason

    def apply(self, game: Game) -> None:
        game.world.block_path(self.location_key, self.direction, self.reason)


class UnblockPathEffect(Effect):
    def __init__(self, location_key: str, direction: str):
        self.location_key = location_key
        self.direction = direction

    def apply(self, game: Game) -> None:
        game.world.unblock_path(self.location_key, self.direction)


class ConditionalEffect(Effect):
    def __init__(
        self,
        *,
        required_flags: set[str] | None = None,
        blocked_flags: set[str] | None = None,
        required_items: list[str] | None = None,
        success_effect: Effect,
        failure_effect: Effect | None = None,
    ):
        self.required_flags = required_flags or set()
        self.blocked_flags = blocked_flags or set()
        self.required_items = required_items or []
        self.success_effect = success_effect
        self.failure_effect = failure_effect

    def apply(self, game: Game) -> None:
        ok = True
        if not self.required_flags.issubset(game.flags):
            ok = False
        if self.blocked_flags.intersection(game.flags):
            ok = False
        if not game.player.has_items(self.required_items):
            ok = False
        if ok:
            self.success_effect.apply(game)
        elif self.failure_effect is not None:
            self.failure_effect.apply(game)


class ChanceEffect(Effect):
    def __init__(
        self,
        chance: float,
        success_effect: Effect,
        failure_effect: Effect | None = None,
    ):
        self.chance = chance
        self.success_effect = success_effect
        self.failure_effect = failure_effect

    def apply(self, game: Game) -> None:
        if random.random() < self.chance:
            self.success_effect.apply(game)
        elif self.failure_effect is not None:
            self.failure_effect.apply(game)


class CompositeEffect(Effect):
    def __init__(self, *effects: Effect):
        self.effects = list(effects)

    def apply(self, game: Game) -> None:
        for effect in self.effects:
            effect.apply(game)


# ============================================================
# Controllers and characters
# ============================================================


class Controller:
    def choose_combat_action(
        self, game: Game, actor: Character, opponent: Character
    ) -> tuple[str, str | None]:
        raise NotImplementedError


class PlayerController(Controller):
    def choose_combat_action(
        self, game: Game, actor: Character, opponent: Character
    ) -> tuple[str, str | None]:
        options = [
            {"kind": "attack", "text": "Attack", "value": None},
            {"kind": "defend", "text": "Defend", "value": None},
            {"kind": "item", "text": "Use consumable", "value": None},
            {"kind": "run", "text": "Run", "value": None},
            {"kind": "surrender", "text": "Surrender", "value": None},
        ]
        choice = game.choose(options, prompt="Choose a combat action: ")
        if choice["kind"] == "item":
            consumables = actor.consumable_items(game.world.item_db)
            if not consumables:
                print("You have no consumables to use.")
                return "none", None
            item_options = []
            for item_id in consumables:
                item = game.world.item(item_id)
                item_options.append(
                    {
                        "kind": "use",
                        "text": f"Use {item.name} (+{item.healing} HP)",
                        "value": item_id,
                    }
                )
            item_options.append({"kind": "back", "text": "Back", "value": None})
            item_choice = game.choose(item_options, prompt="Choose an item: ")
            if item_choice["kind"] == "back":
                return "none", None
            return "item", item_choice["value"]
        return choice["kind"], choice["value"]


class AIController(Controller):
    def choose_combat_action(
        self, game: Game, actor: Character, opponent: Character
    ) -> tuple[str, str | None]:
        consumables = actor.consumable_items(game.world.item_db)
        hp_ratio = actor.health / max(1, actor.total_max_hp(game.world.item_db))
        if consumables and hp_ratio <= 0.35 and random.random() < 0.60:
            return "item", consumables[0]
        if isinstance(actor, NPC) and actor.aggression < 35 and random.random() < 0.25:
            return "defend", None
        return "attack", None


class Character:
    EQUIPMENT_SLOTS = ("weapon", "armor", "charm")

    def __init__(
        self,
        name: str,
        base_stats: Stats,
        controller: Controller,
        *,
        location: str | None = None,
        gold: int = 0,
    ):
        self.name = name
        self.base_stats = base_stats
        self.controller = controller
        self.location = location
        self.health = base_stats.max_hp
        self.gold = gold
        self.inventory: list[str] = []
        self.equipment: dict[str, str | None] = {
            slot: None for slot in self.EQUIPMENT_SLOTS
        }

    def is_alive(self) -> bool:
        return self.health > 0

    def item_counts(self) -> Counter:
        return Counter(self.inventory)

    def has_item(self, item_id: str) -> bool:
        return item_id in self.inventory

    def has_items(self, item_ids: list[str]) -> bool:
        owned = self.item_counts()
        need = Counter(item_ids)
        return all(owned[item_id] >= count for item_id, count in need.items())

    def add_item(self, item_id: str) -> None:
        self.inventory.append(item_id)

    def add_items(self, item_ids: list[str]) -> None:
        for item_id in item_ids:
            self.add_item(item_id)

    def remove_item(self, item_id: str) -> bool:
        if item_id in self.inventory:
            self.inventory.remove(item_id)
            for slot, equipped in self.equipment.items():
                if equipped == item_id:
                    self.equipment[slot] = None
            return True
        return False

    def remove_items(self, item_ids: list[str]) -> bool:
        if not self.has_items(item_ids):
            return False
        for item_id in item_ids:
            self.remove_item(item_id)
        return True

    def equipped_items(self, item_db: dict[str, Item]) -> list[Item]:
        return [
            item_db[item_id]
            for item_id in self.equipment.values()
            if item_id is not None
        ]

    def total_max_hp(self, item_db: dict[str, Item]) -> int:
        return self.base_stats.max_hp + sum(
            item.hp_bonus for item in self.equipped_items(item_db)
        )

    def total_defense(self, item_db: dict[str, Item]) -> int:
        return self.base_stats.defense + sum(
            item.defense_bonus for item in self.equipped_items(item_db)
        )

    def attack_range(self, item_db: dict[str, Item]) -> tuple[int, int]:
        bonus = sum(item.power_bonus for item in self.equipped_items(item_db))
        return self.base_stats.attack_min + bonus, self.base_stats.attack_max + bonus

    def roll_attack_damage(self, item_db: dict[str, Item]) -> int:
        low, high = self.attack_range(item_db)
        return random.randint(low, high)

    def take_damage(self, raw_damage: int, item_db: dict[str, Item]) -> int:
        actual = max(1, raw_damage - self.total_defense(item_db))
        self.health = max(0, self.health - actual)
        return actual

    def heal(self, amount: int, item_db: dict[str, Item]) -> int:
        before = self.health
        self.health = min(self.total_max_hp(item_db), self.health + amount)
        return self.health - before

    def clamp_health(self, item_db: dict[str, Item]) -> None:
        self.health = min(self.health, self.total_max_hp(item_db))

    def equippable_items(self, item_db: dict[str, Item]) -> list[str]:
        result = []
        seen = set()
        for item_id in self.inventory:
            if item_id in seen:
                continue
            seen.add(item_id)
            if item_db[item_id].slot in self.EQUIPMENT_SLOTS:
                result.append(item_id)
        return result

    def consumable_items(self, item_db: dict[str, Item]) -> list[str]:
        result = []
        seen = set()
        for item_id in self.inventory:
            if item_id in seen:
                continue
            seen.add(item_id)
            if item_db[item_id].healing > 0:
                result.append(item_id)
        return result

    def equip(self, item_id: str, item_db: dict[str, Item]) -> tuple[bool, str]:
        if item_id not in self.inventory:
            return False, f"{self.name} does not have that item."
        item = item_db[item_id]
        if item.slot not in self.EQUIPMENT_SLOTS:
            return False, f"{item.name} cannot be equipped."
        previous = self.equipment[item.slot]
        self.equipment[item.slot] = item_id
        self.clamp_health(item_db)
        if previous is None:
            return True, f"Equipped {item.name} in {item.slot} slot."
        return True, f"Equipped {item.name}, replacing {item_db[previous].name}."

    def unequip(self, slot: str, item_db: dict[str, Item]) -> tuple[bool, str]:
        if slot not in self.EQUIPMENT_SLOTS:
            return False, "Invalid equipment slot."
        current = self.equipment[slot]
        if current is None:
            return False, f"Nothing is equipped in {slot}."
        item_name = item_db[current].name
        self.equipment[slot] = None
        self.clamp_health(item_db)
        return True, f"Unequipped {item_name}."

    def use_consumable(
        self, item_id: str, item_db: dict[str, Item]
    ) -> tuple[bool, str, int]:
        if item_id not in self.inventory:
            return False, "Item not in inventory.", 0
        item = item_db[item_id]
        if item.healing <= 0:
            return False, f"{item.name} is not a consumable healing item.", 0
        self.remove_item(item_id)
        healed = self.heal(item.healing, item_db)
        return True, f"Used {item.name}.", healed

    def show_status(self, item_db: dict[str, Item], *, bounty: int = 0) -> None:
        low, high = self.attack_range(item_db)
        divider("STATUS")
        print(f"Name: {self.name}")
        print(f"Health: {self.health}/{self.total_max_hp(item_db)}")
        print(f"Attack: {low}-{high}")
        print(f"Defense: {self.total_defense(item_db)}")
        print(f"Gold: {self.gold}")
        print(f"Bounty: {bounty}")
        print("\nEquipment:")
        for slot in self.EQUIPMENT_SLOTS:
            item_id = self.equipment[slot]
            label = item_db[item_id].name if item_id is not None else "empty"
            print(f"- {slot}: {label}")
        print("\nInventory:")
        if not self.inventory:
            print("- empty")
        else:
            counts = Counter(self.inventory)
            seen = set()
            for item_id in self.inventory:
                if item_id in seen:
                    continue
                seen.add(item_id)
                suffix = f" x{counts[item_id]}" if counts[item_id] > 1 else ""
                print(f"- {item_db[item_id].name}{suffix}")


# ============================================================
# Shared NPC model
# ============================================================


class NPC(Character):
    def __init__(
        self,
        name: str,
        base_stats: Stats,
        *,
        description: str = "",
        tags: set[str] | None = None,
        aggression: int = 0,
        courage: int = 50,
        willingness_to_trade: int = 0,
        hostile: bool = False,
        dialogue_topics: list[DialogueTopic] | None = None,
        trade_offers: list[TradeOffer] | None = None,
        riddle: Riddle | None = None,
        surrender_at_ratio: float | None = None,
        surrender_lines: list[str] | None = None,
        surrender_trade_offers: list[TradeOffer] | None = None,
        peaceful_after_surrender: bool = True,
        surrender_accept_lines: list[str] | None = None,
        surrender_reject_lines: list[str] | None = None,
        surrender_accept_effect: Effect | None = None,
        defeat_lines: list[str] | None = None,
        reward_items: list[str] | None = None,
        reward_flags: set[str] | None = None,
        reward_gold: int = 0,
        persistent: bool = True,
        inventory: list[str] | None = None,
        equipment: dict[str, str | None] | None = None,
        gold: int = 0,
    ):
        super().__init__(
            name=name, base_stats=base_stats, controller=AIController(), gold=gold
        )
        self.description = description
        self.tags = tags or set()
        self.aggression = aggression
        self.courage = courage
        self.willingness_to_trade = willingness_to_trade
        self.hostile = hostile
        self.dialogue_topics = dialogue_topics or []
        self.trade_offers = trade_offers or []
        self.riddle = riddle
        self.surrender_at_ratio = surrender_at_ratio
        self.surrender_lines = surrender_lines or []
        self.surrender_trade_offers = surrender_trade_offers or []
        self.peaceful_after_surrender = peaceful_after_surrender
        self.surrender_accept_lines = surrender_accept_lines or []
        self.surrender_reject_lines = surrender_reject_lines or []
        self.surrender_accept_effect = surrender_accept_effect
        self.defeat_lines = defeat_lines or []
        self.reward_items = reward_items or []
        self.reward_flags = reward_flags or set()
        self.reward_gold = reward_gold
        self.persistent = persistent
        self.used_topics: set[str] = set()
        self.completed_trades: set[int] = set()
        self.surrendered = False
        self.riddle_solved = False
        self.defeated = False

        for item_id in inventory or []:
            self.add_item(item_id)
        if equipment:
            for slot, item_id in equipment.items():
                self.equipment[slot] = item_id

    def is_guard(self) -> bool:
        return "guard" in self.tags or "law" in self.tags

    def mood_label(self) -> str:
        if self.defeated:
            return "defeated"
        if self.surrendered:
            return "surrendered"
        if self.hostile:
            return "hostile"
        if self.willingness_to_trade >= 70:
            return "open to trade"
        if self.aggression >= 60:
            return "dangerous"
        return "neutral"

    def menu_text(self) -> str:
        return f"Approach {self.name}"

    def available_topics(self, game: Game) -> list[DialogueTopic]:
        results = []
        for topic in self.dialogue_topics:
            if topic.once and topic.key in self.used_topics:
                continue
            if not topic.required_flags.issubset(game.flags):
                continue
            if topic.blocked_flags.intersection(game.flags):
                continue
            if not game.player.has_items(topic.required_items):
                continue
            results.append(topic)
        return results

    def available_trade_offers(self, game: Game) -> list[tuple[int, TradeOffer]]:
        results = []
        for index, offer in enumerate(self.trade_offers):
            if not offer.repeatable and index in self.completed_trades:
                continue
            if not offer.required_flags.issubset(game.flags):
                continue
            if offer.blocked_flags.intersection(game.flags):
                continue
            if self.gold < offer.gives_gold:
                continue
            if not self.has_items(offer.gives_items):
                continue
            results.append((index, offer))
        return results

    def can_talk(self, game: Game) -> bool:
        return self.is_alive() and bool(self.available_topics(game))

    def can_trade(self, game: Game) -> bool:
        if not self.is_alive():
            return False
        if not self.available_trade_offers(game):
            return False
        if self.surrendered:
            return True
        return self.willingness_to_trade >= 30 and not self.hostile

    def can_riddle(self) -> bool:
        return self.is_alive() and self.riddle is not None and not self.riddle_solved

    def can_attack(self) -> bool:
        return self.is_alive()

    def can_loot(self) -> bool:
        return self.defeated and (bool(self.inventory) or self.gold > 0)

    def on_player_attack(self, game: Game) -> None:
        if not self.hostile and not self.defeated:
            if self.is_guard():
                ChangeBountyEffect(75, f"Assaulting guard {self.name}").apply(game)
            elif self.aggression < 50:
                ChangeBountyEffect(25, f"Assaulting {self.name}").apply(game)
        self.hostile = True
        self.aggression = max(self.aggression, 70)

    def maybe_surrender(self, game: Game) -> bool:
        if self.surrendered or self.surrender_at_ratio is None or not self.is_alive():
            return False
        ratio = self.health / max(1, self.total_max_hp(game.world.item_db))
        if ratio > self.surrender_at_ratio:
            return False
        self.surrendered = True
        if self.peaceful_after_surrender:
            self.hostile = False
        self.willingness_to_trade = max(self.willingness_to_trade, 80)
        self.trade_offers.extend(self.surrender_trade_offers)
        for line in self.surrender_lines:
            say(self.name, line)
        return True

    def will_accept_player_surrender(self, game: Game) -> bool:
        if self.defeated or not self.is_alive():
            return False
        if self.is_guard():
            return True
        if self.surrendered:
            return True
        return self.aggression < 90 or self.willingness_to_trade > 40

    def handle_player_surrender(self, game: Game) -> bool:
        if not self.will_accept_player_surrender(game):
            lines = self.surrender_reject_lines or ["No. This ends here."]
            for line in lines:
                say(self.name, line)
            self.hostile = True
            return False

        lines = self.surrender_accept_lines or ["Drop your guard and back away slowly."]
        for line in lines:
            say(self.name, line)

        self.hostile = False
        if self.surrender_accept_effect is not None:
            self.surrender_accept_effect.apply(game)
        else:
            if self.aggression >= 70:
                if game.player.gold > 0:
                    stolen = min(game.player.gold, max(5, game.player.gold // 2))
                    game.player.gold -= stolen
                    self.gold += stolen
                    print(f"{self.name} takes {stolen} gold from you.")
                elif game.player.inventory:
                    RemoveRandomItemEffect(text_prefix=f"{self.name} takes").apply(game)
            game.enter_mode("exploration")
        if game.mode == "combat":
            game.enter_mode("exploration")
        return True

    def on_defeat(self, game: Game) -> None:
        if self.defeated:
            return
        self.defeated = True
        self.hostile = False
        self.health = 0

        if self.surrendered:
            ChangeBountyEffect(100, f"Killing surrendering {self.name}").apply(game)
        elif self.is_guard():
            ChangeBountyEffect(150, f"Killing guard {self.name}").apply(game)
        elif self.aggression < 50:
            ChangeBountyEffect(40, f"Killing non-hostile {self.name}").apply(game)

        for line in self.defeat_lines:
            print(line)
        for item_id in self.reward_items:
            game.give_player_item(item_id)
        if self.reward_gold:
            game.player.gold += self.reward_gold
            print(f"You gain {self.reward_gold} gold.")
        for flag in self.reward_flags:
            game.flags.add(flag)
        game.world.on_npc_defeated(game, self)


class MerchantNPC(NPC):
    def __init__(self, name: str, base_stats: Stats | None = None, **kwargs):
        kwargs.setdefault("aggression", 10)
        kwargs.setdefault("courage", 40)
        kwargs.setdefault("willingness_to_trade", 90)
        kwargs.setdefault("hostile", False)
        super().__init__(
            name=name, base_stats=base_stats or Stats(35, 4, 6, 1), **kwargs
        )


class BanditNPC(NPC):
    def __init__(self, name: str, base_stats: Stats | None = None, **kwargs):
        kwargs.setdefault("aggression", 70)
        kwargs.setdefault("courage", 35)
        kwargs.setdefault("willingness_to_trade", 10)
        kwargs.setdefault("hostile", False)
        kwargs.setdefault("surrender_at_ratio", 0.25)
        super().__init__(
            name=name, base_stats=base_stats or Stats(28, 5, 9, 1), **kwargs
        )


class GuardianNPC(NPC):
    def __init__(self, name: str, base_stats: Stats | None = None, **kwargs):
        kwargs.setdefault("aggression", 50)
        kwargs.setdefault("courage", 90)
        kwargs.setdefault("willingness_to_trade", 0)
        kwargs.setdefault("hostile", False)
        super().__init__(
            name=name, base_stats=base_stats or Stats(38, 7, 11, 2), **kwargs
        )


class ElderNPC(NPC):
    def __init__(self, name: str, base_stats: Stats | None = None, **kwargs):
        kwargs.setdefault("aggression", 5)
        kwargs.setdefault("courage", 20)
        kwargs.setdefault("willingness_to_trade", 15)
        kwargs.setdefault("hostile", False)
        super().__init__(
            name=name, base_stats=base_stats or Stats(40, 4, 7, 0), **kwargs
        )


class GuardNPC(NPC):
    def __init__(self, name: str, base_stats: Stats | None = None, **kwargs):
        kwargs.setdefault("tags", {"guard", "law"})
        kwargs.setdefault("aggression", 40)
        kwargs.setdefault("courage", 85)
        kwargs.setdefault("willingness_to_trade", 0)
        kwargs.setdefault("hostile", False)
        super().__init__(
            name=name, base_stats=base_stats or Stats(45, 6, 9, 2), **kwargs
        )


# ============================================================
# Features / world objects
# ============================================================


class Feature:
    def __init__(self, name: str, verb: str = "Inspect"):
        self.name = name
        self.verb = verb

    def menu_text(self) -> str:
        return f"{self.verb} {self.name}"

    def interact(self, game: Game) -> None:
        print(f"Nothing happens when you interact with {self.name}.")


class TextFeature(Feature):
    def __init__(self, name: str, text: str, verb: str = "Inspect"):
        super().__init__(name=name, verb=verb)
        self.text = text

    def interact(self, game: Game) -> None:
        print("\n" + self.text)


class ScriptedFeature(Feature):
    def __init__(
        self,
        name: str,
        *,
        verb: str = "Use",
        required_flags: set[str] | None = None,
        blocked_text: str = "Nothing happens.",
        once_flag: str | None = None,
        first_effect: Effect | None = None,
        repeat_effect: Effect | None = None,
    ):
        super().__init__(name=name, verb=verb)
        self.required_flags = required_flags or set()
        self.blocked_text = blocked_text
        self.once_flag = once_flag
        self.first_effect = first_effect
        self.repeat_effect = repeat_effect

    def interact(self, game: Game) -> None:
        if not self.required_flags.issubset(game.flags):
            print(self.blocked_text)
            return
        if self.once_flag and self.once_flag in game.flags:
            if self.repeat_effect is not None:
                self.repeat_effect.apply(game)
            else:
                print("Nothing more happens.")
            return
        if self.first_effect is not None:
            self.first_effect.apply(game)
        if self.once_flag:
            game.flags.add(self.once_flag)
        if self.once_flag is None and self.repeat_effect is not None:
            self.repeat_effect.apply(game)


# ============================================================
# Encounters
# ============================================================


class EncounterRule:
    def __init__(
        self,
        name: str,
        *,
        locations: set[str],
        chance: float,
        handler: Callable[[Game], bool],
        required_flags: set[str] | None = None,
        blocked_flags: set[str] | None = None,
        once_flag: str | None = None,
        predicate: Callable[[Game], bool] | None = None,
    ):
        self.name = name
        self.locations = locations
        self.chance = chance
        self.handler = handler
        self.required_flags = required_flags or set()
        self.blocked_flags = blocked_flags or set()
        self.once_flag = once_flag
        self.predicate = predicate

    def try_trigger(self, game: Game) -> bool:
        location = game.current_location()
        if location.key not in self.locations:
            return False
        if self.once_flag and self.once_flag in game.flags:
            return False
        if not self.required_flags.issubset(game.flags):
            return False
        if self.blocked_flags.intersection(game.flags):
            return False
        if self.predicate is not None and not self.predicate(game):
            return False
        if random.random() > self.chance:
            return False
        triggered = self.handler(game)
        if triggered and self.once_flag:
            game.flags.add(self.once_flag)
        return triggered


# ============================================================
# World base class
# ============================================================


class World:
    def __init__(self, name: str):
        self.name = name
        self.locations: dict[str, Location] = {}
        self.item_db: dict[str, Item] = {}
        self.encounter_rules: list[EncounterRule] = []

    def define_item(self, item: Item) -> None:
        self.item_db[item.item_id] = item

    def item(self, item_id: str) -> Item:
        return self.item_db[item_id]

    def add_location(self, key: str, name: str, description: str) -> None:
        self.locations[key] = Location(key=key, name=name, description=description)

    def get(self, key: str) -> Location:
        return self.locations[key]

    def connect(
        self,
        source: str,
        direction: str,
        destination: str,
        *,
        requires_item: str | None = None,
        requires_flag: str | None = None,
        blocked: bool = False,
        blocked_text: str = "That path is blocked.",
        warning_text: str | None = None,
        on_attempt_effect: Effect | None = None,
        on_blocked_effect: Effect | None = None,
        on_success_effect: Effect | None = None,
    ) -> None:
        self.locations[source].exits.append(
            Exit(
                direction=direction,
                destination=destination,
                requires_item=requires_item,
                requires_flag=requires_flag,
                blocked=blocked,
                blocked_text=blocked_text,
                warning_text=warning_text,
                on_attempt_effect=on_attempt_effect,
                on_blocked_effect=on_blocked_effect,
                on_success_effect=on_success_effect,
            )
        )

    def connect_two_way(
        self, a: str, dir_a_to_b: str, b: str, dir_b_to_a: str, **kwargs
    ) -> None:
        self.connect(a, dir_a_to_b, b, **kwargs)
        self.connect(b, dir_b_to_a, a)

    def place_item(self, location_key: str, item_id: str) -> None:
        self.locations[location_key].items.append(item_id)

    def place_feature(self, location_key: str, feature: Feature) -> None:
        self.locations[location_key].features.append(feature)

    def place_npc(self, location_key: str, npc: NPC) -> None:
        npc.location = location_key
        self.locations[location_key].npcs.append(npc)

    def add_encounter_rule(self, rule: EncounterRule) -> None:
        self.encounter_rules.append(rule)

    def find_exit(self, location_key: str, direction: str) -> Exit | None:
        for exit_obj in self.locations[location_key].exits:
            if exit_obj.direction == direction:
                return exit_obj
        return None

    def block_path(self, location_key: str, direction: str, reason: str) -> None:
        exit_obj = self.find_exit(location_key, direction)
        if exit_obj is not None:
            exit_obj.blocked = True
            exit_obj.blocked_text = reason

    def unblock_path(self, location_key: str, direction: str) -> None:
        exit_obj = self.find_exit(location_key, direction)
        if exit_obj is not None:
            exit_obj.blocked = False

    def on_location_enter(self, game: Game, location: Location) -> None:
        pass

    def on_item_picked_up(self, game: Game, item_id: str) -> None:
        pass

    def on_npc_defeated(self, game: Game, npc: NPC) -> None:
        pass

    def try_start_of_exploration_encounter(self, game: Game) -> bool:
        for rule in self.encounter_rules:
            if rule.try_trigger(game):
                return True
        return False


# ============================================================
# Concrete world
# ============================================================


class StarCrystalWorld(World):
    def __init__(self):
        super().__init__(name="The Star Crystal")
        self.build_items()
        self.build_locations()
        self.build_paths()
        self.build_features()
        self.build_npcs()
        self.scatter_items()
        self.build_encounters()

    def build_items(self) -> None:
        self.define_item(
            Item("village_map", "Village Map", "A hand-drawn map of the valley.")
        )
        self.define_item(
            Item(
                "moonleaf_herb",
                "Moonleaf Herb",
                "A silver-blue herb that smells sharp and fresh.",
            )
        )
        self.define_item(
            Item("rope", "Rope", "Strong rope left behind at an old campsite.")
        )
        self.define_item(
            Item("lantern", "Lantern", "A sturdy lantern. Essential in dark places.")
        )
        self.define_item(
            Item(
                "silver_key",
                "Silver Key",
                "A polished key shaped like a crescent moon.",
            )
        )
        self.define_item(
            Item(
                "star_crystal",
                "Star Crystal",
                "A warm crystal filled with light.",
                set_flags_on_pickup={"has_star_crystal"},
            )
        )

        self.define_item(
            Item(
                "iron_sword",
                "Iron Sword",
                "Rusty, but still dependable.",
                slot="weapon",
                power_bonus=4,
            )
        )
        self.define_item(
            Item(
                "guard_coat",
                "Guard Coat",
                "Reinforced cloth armor.",
                slot="armor",
                defense_bonus=2,
                hp_bonus=10,
            )
        )
        self.define_item(
            Item(
                "lucky_coin",
                "Lucky Coin",
                "A coin etched with a star. It makes you strangely steady.",
                slot="charm",
                defense_bonus=1,
            )
        )
        self.define_item(
            Item(
                "hunter_spear",
                "Hunter Spear",
                "A balanced spear made to keep biting creatures at a distance.",
                slot="weapon",
                power_bonus=6,
            )
        )

        self.define_item(
            Item(
                "health_tonic",
                "Health Tonic",
                "A red tonic that restores health.",
                healing=20,
                tags={"consumable"},
            )
        )
        self.define_item(
            Item(
                "greater_tonic",
                "Greater Tonic",
                "A stronger restorative draught.",
                healing=35,
                tags={"consumable"},
            )
        )

    def build_locations(self) -> None:
        self.add_location(
            "village",
            "Sunmeadow Village",
            "A quiet village square surrounds a dry fountain. Traders, guards, and villagers drift across the square.",
        )
        self.add_location(
            "crossroads",
            "Old Crossroads",
            "Four dirt paths meet under a leaning wooden sign.",
        )
        self.add_location(
            "forest",
            "Whispering Forest",
            "Tall trees sway overhead. Moss, roots, and old camp supplies lie under the leaves.",
        )
        self.add_location(
            "garden",
            "Hidden Garden",
            "A secret clearing full of bright flowers and soft green light.",
        )
        self.add_location(
            "lake",
            "Mirror Lake",
            "A glassy lake reflects the sky. A weathered fisherman waits by the dock.",
        )
        self.add_location(
            "cave_entrance",
            "Cave Entrance",
            "A crack in the hillside opens into darkness. Cold air spills out.",
        )
        self.add_location(
            "cave_depths",
            "Crystal Cave",
            "Blue crystals glow faintly from the cave walls.",
        )
        self.add_location(
            "ruins",
            "Moonstone Ruins",
            "Broken pillars and ancient arches rise from a hilltop.",
        )
        self.add_location(
            "tower_gate",
            "Tower Gate",
            "An iron gate blocks the tower stairs. The keyhole is shaped like a crescent moon.",
        )
        self.add_location(
            "tower_top",
            "Tower Summit",
            "Wind curls around the broken top of the tower. A glowing crystal rests on a pedestal.",
        )
        self.add_location(
            "jail",
            "Village Jail",
            "A cramped stone cell. The door is locked from the outside, and there is no key within reach.",
        )

    def build_paths(self) -> None:
        self.connect_two_way("village", "north", "crossroads", "south")
        self.connect_two_way("crossroads", "west", "forest", "east")
        self.connect_two_way("crossroads", "east", "lake", "west")

        self.connect(
            "crossroads",
            "north",
            "ruins",
            on_success_effect=ConditionalEffect(
                blocked_flags={"ruins_ambush_triggered"},
                success_effect=CompositeEffect(
                    PrintEffect(
                        "As you head toward the ruins, a loose stone gives way underfoot and you stumble."
                    ),
                    DamageCharacterEffect("player", 4),
                    SetFlagEffect("ruins_ambush_triggered"),
                ),
            ),
        )
        self.connect("ruins", "south", "crossroads")

        self.connect_two_way("forest", "west", "garden", "east")

        self.connect(
            "forest",
            "north",
            "cave_entrance",
            warning_text="The bramble path ahead looks painful. Push through anyway?",
            on_success_effect=ChanceEffect(
                0.60,
                CompositeEffect(
                    PrintEffect(
                        "Thorns rake your arms and tear at your clothes as you force your way through."
                    ),
                    DamageCharacterEffect("player", 6),
                ),
                PrintEffect(
                    "You find a careful route through the brambles and avoid the worst of them."
                ),
            ),
        )
        self.connect("cave_entrance", "south", "forest")

        self.connect(
            "cave_entrance",
            "north",
            "cave_depths",
            requires_item="lantern",
            blocked_text="It is too dark to enter safely. You need a Lantern.",
            on_blocked_effect=PrintEffect(
                "Shadows shift in the crack ahead. You reconsider."
            ),
        )
        self.connect("cave_depths", "south", "cave_entrance")

        self.connect(
            "ruins",
            "north",
            "tower_gate",
            requires_item="rope",
            blocked_text="The path ahead is broken. You need a Rope to cross safely.",
            warning_text="The crossing is narrow and unstable. Go on?",
            on_success_effect=ChanceEffect(
                0.30,
                CompositeEffect(
                    PrintEffect(
                        "The old crossing gives way beneath you for a moment. Your pack slams against the stone."
                    ),
                    DamageCharacterEffect("player", 5),
                    ConditionalEffect(
                        required_items=["health_tonic"],
                        success_effect=RemoveItemEffect(
                            "health_tonic",
                            text="One of your Health Tonics shatters on the rocks.",
                        ),
                    ),
                ),
                PrintEffect("You cross the broken path without incident."),
            ),
            on_blocked_effect=PrintEffect(
                "The gap ahead looks fatal without proper support."
            ),
        )
        self.connect("tower_gate", "south", "ruins")

        self.connect(
            "tower_gate",
            "up",
            "tower_top",
            requires_item="silver_key",
            blocked_text="The gate is locked. You need the Silver Key.",
            on_success_effect=ConditionalEffect(
                blocked_flags={"tower_entry_event"},
                success_effect=CompositeEffect(
                    PrintEffect(
                        "Ancient light flickers along the stairwell as the gate opens."
                    ),
                    SetFlagEffect("tower_entry_event"),
                ),
            ),
        )
        self.connect("tower_top", "down", "tower_gate")

    def build_features(self) -> None:
        self.place_feature(
            "village",
            ScriptedFeature(
                name="fountain",
                verb="Rest at",
                first_effect=HealPlayerEffect(
                    amount=15,
                    heal_text="You sit beside the fountain and catch your breath.",
                    full_text="You already feel fully rested.",
                ),
            ),
        )
        self.place_feature(
            "crossroads",
            TextFeature(
                name="wooden sign",
                text=(
                    "NORTH -> Moonstone Ruins\n"
                    "EAST  -> Mirror Lake\n"
                    "WEST  -> Whispering Forest\n"
                    "SOUTH -> Sunmeadow Village"
                ),
                verb="Read",
            ),
        )
        self.place_feature(
            "forest",
            ScriptedFeature(
                name="whispering trees",
                verb="Listen to",
                once_flag="forest_hint_seen",
                first_effect=PrintEffect(
                    "A cool breeze seems to whisper: 'Light opens the dark path.'"
                ),
                repeat_effect=PrintEffect(
                    "The leaves rustle softly. The forest has already shared its hint."
                ),
            ),
        )
        self.place_feature(
            "forest",
            ScriptedFeature(
                name="unstable hillside",
                verb="Climb",
                once_flag="landslide_happened",
                first_effect=CompositeEffect(
                    PrintEffect(
                        [
                            "You scramble up the unstable slope.",
                            "A deep crack echoes through the trees.",
                            "Rocks crash down and bury the garden trail!",
                        ]
                    ),
                    BlockPathEffect(
                        "forest",
                        "west",
                        "A landslide has buried the trail under loose rocks.",
                    ),
                    BlockPathEffect(
                        "garden",
                        "east",
                        "A landslide has buried the trail under loose rocks.",
                    ),
                ),
                repeat_effect=PrintEffect(
                    "The hillside already collapsed. Rocks still cover the trail."
                ),
            ),
        )
        self.place_feature(
            "ruins",
            ScriptedFeature(
                name="old winch",
                verb="Turn",
                required_flags={"landslide_happened"},
                blocked_text="The mechanism is tied to the collapsed trail, but nothing needs clearing yet.",
                once_flag="garden_path_reopened",
                first_effect=CompositeEffect(
                    PrintEffect(
                        [
                            "You turn the ancient winch.",
                            "Some buried stones shift and part of the trail is cleared.",
                        ]
                    ),
                    UnblockPathEffect("forest", "west"),
                    UnblockPathEffect("garden", "east"),
                ),
                repeat_effect=PrintEffect("The old winch has already done all it can."),
            ),
        )
        self.place_feature(
            "cave_depths",
            TextFeature(
                name="glowing crystals",
                text="Their light feels ancient, calm, and strangely familiar. You begin to suspect the tower and village are connected by old magic.",
                verb="Study",
            ),
        )
        self.place_feature(
            "jail",
            ScriptedFeature(
                name="prison cot",
                verb="Serve time on",
                required_flags={"jailed"},
                blocked_text="There is no sentence to serve right now.",
                first_effect=CompositeEffect(
                    PrintEffect(
                        [
                            "You sit on the narrow cot and wait.",
                            "Hours pass. Your anger cools. The guard finally opens the cell.",
                        ]
                    ),
                    SetBountyEffect(0),
                    ClearFlagEffect("jailed"),
                    MovePlayerEffect(
                        "village", text="You are released back into Sunmeadow Village."
                    ),
                    HealPlayerEffect(
                        999,
                        heal_text="A night's rest mends your wounds.",
                        full_text="You are already fully rested.",
                    ),
                ),
            ),
        )

    def build_npcs(self) -> None:
        elder_topics = [
            DialogueTopic(
                key="dry_fountain",
                title="Ask about the dry fountain",
                lines=[
                    "Our fountain has run dry, and the tower's shadow grows longer every night.",
                    "Bring back the Star Crystal, and Sunmeadow may shine again.",
                ],
                blocked_flags={"quest_started"},
                once=True,
                outcome_effect=CompositeEffect(
                    GivePlayerItemEffect("village_map"),
                    SetFlagEffect("quest_started"),
                ),
            ),
            DialogueTopic(
                key="advice",
                title="Ask what to do next",
                lines=[
                    "Listen to the valley. The people, the ruins, and even the trees know more than they first reveal."
                ],
                required_flags={"quest_started"},
                blocked_flags={"has_star_crystal", "game_won"},
            ),
            DialogueTopic(
                key="return_crystal",
                title="Return the Star Crystal",
                lines=[
                    "You found it... the Star Crystal!",
                    "You raise the crystal above the fountain.",
                    "Light pours across the square. Water bursts upward. Cheers echo through the village.",
                    "You saved Sunmeadow Village. You win!",
                ],
                required_flags={"has_star_crystal"},
                blocked_flags={"game_won"},
                once=True,
                outcome_effect=SetFlagEffect("game_won"),
            ),
        ]
        self.place_npc(
            "village",
            ElderNPC(
                "Elder Mira",
                description="The village elder watches the dry fountain with worried eyes.",
                dialogue_topics=elder_topics,
                inventory=["health_tonic"],
                defeat_lines=["Elder Mira falls. The square goes silent."],
            ),
        )

        self.place_npc(
            "village",
            GuardNPC(
                "Guard Halwen",
                description="A village guard in a faded coat watches the square with practiced patience.",
                dialogue_topics=[
                    DialogueTopic(
                        key="guard_law",
                        title="Ask about the law",
                        lines=[
                            "Keep the peace, pay your debts, and do not test my patience.",
                            "If you draw steel on the innocent, I will answer it.",
                        ],
                    )
                ],
                surrender_accept_lines=[
                    "Drop your weapon and come quietly.",
                    "You can explain yourself from the cell.",
                ],
                surrender_accept_effect=CompositeEffect(
                    SetFlagEffect("jailed"),
                    MovePlayerEffect(
                        "jail",
                        text="Guard Halwen binds your hands and marches you to the village jail.",
                    ),
                ),
                inventory=["guard_coat", "health_tonic"],
                equipment={"armor": "guard_coat"},
                defeat_lines=[
                    "Guard Halwen crashes to the stones. A hush spreads across the square."
                ],
                gold=12,
            ),
        )

        self.place_npc(
            "village",
            MerchantNPC(
                "Merchant Sella",
                description="A sharp-eyed merchant has laid out weapons, tonics, and travel gear on a thick canvas cloth.",
                dialogue_topics=[
                    DialogueTopic(
                        key="merchant_pitch",
                        title="Ask what she recommends",
                        lines=[
                            "If you are heading for caves, buy reach and buy confidence.",
                            "Spiders hate a spear point more than a short blade.",
                        ],
                    )
                ],
                trade_offers=[
                    TradeOffer(
                        title="Buy Hunter Spear for 12 gold",
                        wants_gold=12,
                        gives_items=["hunter_spear"],
                    ),
                    TradeOffer(
                        title="Buy Health Tonic for 6 gold",
                        wants_gold=6,
                        gives_items=["health_tonic"],
                        repeatable=True,
                    ),
                ],
                inventory=["hunter_spear", "health_tonic", "health_tonic"],
                gold=30,
            ),
        )

        fisher_topics = [
            DialogueTopic(
                key="lake_gossip",
                title="Ask about the lake",
                lines=[
                    "The lake tells you things if you wait long enough.",
                    "Mostly warnings. Sometimes luck.",
                ],
            ),
            DialogueTopic(
                key="ask_tower",
                title="Ask about the tower",
                lines=[
                    "Dark places do not stay dark forever. Sometimes they only wait for the right light."
                ],
            ),
        ]
        fisher_offers = [
            TradeOffer(
                "Trade Moonleaf Herb for Lantern",
                wants_items=["moonleaf_herb"],
                gives_items=["lantern"],
            ),
            TradeOffer(
                "Trade Lucky Coin for Guard Coat",
                wants_items=["lucky_coin"],
                gives_items=["guard_coat"],
            ),
        ]
        self.place_npc(
            "lake",
            MerchantNPC(
                "Fisher Rowan",
                description="A patient fisherman sits on the dock, humming to himself.",
                dialogue_topics=fisher_topics,
                trade_offers=fisher_offers,
                inventory=["lantern", "guard_coat", "health_tonic"],
                equipment={"armor": "guard_coat"},
                defeat_lines=[
                    "Fisher Rowan drops his rod and backs away from the dock."
                ],
                gold=8,
            ),
        )

        bandit_topics = [
            DialogueTopic(
                key="bandit_warning",
                title="Ask why he is here",
                lines=[
                    "Because hidden places collect hidden things.",
                    "And hidden things can be sold.",
                ],
                blocked_flags={"bandit_spared"},
            ),
            DialogueTopic(
                key="bandit_after_surrender",
                title="Ask what changed his mind",
                lines=["Turns out survival is worth more than pride."],
                required_flags={"bandit_spared"},
            ),
        ]
        bandit_surrender_offers = [
            TradeOffer(
                "Trade Moonleaf Herb for Lucky Coin",
                wants_items=["moonleaf_herb"],
                gives_items=["lucky_coin"],
            ),
            TradeOffer(
                "Trade Rope for Greater Tonic",
                wants_items=["rope"],
                gives_items=["greater_tonic"],
            ),
        ]
        self.place_npc(
            "garden",
            BanditNPC(
                "Bandit Nox",
                description="A lean bandit lurks among the flowers, guarding a hidden stash.",
                dialogue_topics=bandit_topics,
                surrender_lines=[
                    "Wait! Enough! I do not want to die here.",
                    "If you spare me, we can make a deal.",
                ],
                surrender_trade_offers=bandit_surrender_offers,
                surrender_accept_lines=[
                    "Fine. Take what you came for and let me live."
                ],
                peaceful_after_surrender=True,
                inventory=["lucky_coin", "greater_tonic", "health_tonic"],
                equipment={"weapon": "iron_sword"},
                defeat_lines=[
                    "Bandit Nox drops his bag and limps away through the garden."
                ],
                reward_flags={"bandit_defeated"},
                reward_gold=10,
                gold=15,
            ),
        )

        self.place_npc(
            "cave_depths",
            NPC(
                name="Crystal Spider",
                base_stats=Stats(max_hp=24, attack_min=5, attack_max=8, defense=2),
                description="A giant spider clings to a web of glowing crystal threads.",
                tags={"monster"},
                aggression=90,
                courage=100,
                willingness_to_trade=0,
                hostile=False,
                surrender_reject_lines=[
                    "The spider only clicks its mandibles and closes in."
                ],
                inventory=["silver_key", "health_tonic"],
                defeat_lines=["The Crystal Spider curls up and stops moving."],
                reward_flags={"spider_defeated"},
                reward_gold=14,
            ),
        )

        guardian_topics = [
            DialogueTopic(
                key="guardian_warning",
                title="Ask who the crystal belongs to",
                lines=[
                    "The crystal does not belong to one hand. It belongs to balance."
                ],
                blocked_flags={"guardian_passed"},
            ),
            DialogueTopic(
                key="guardian_after_trial",
                title="Ask if the trial is complete",
                lines=["The trial is complete. The Star Crystal may now be taken."],
                required_flags={"guardian_passed"},
            ),
        ]
        guardian_riddle = Riddle(
            intro_lines=["Answer my riddle, and the crystal may be yours."],
            question="I am always running, but I never walk. I often murmur, but I never talk. What am I?",
            answers=["river", "a river", "stream", "a stream"],
            success_lines=["Wisdom and patience walk beside you. Step forward."],
            failure_lines=["Not quite. Think more carefully."],
            repeat_lines=["The trial is complete. The Star Crystal may now be taken."],
            damage_on_failure=12,
            set_flags_on_success={"guardian_passed"},
        )
        self.place_npc(
            "tower_top",
            GuardianNPC(
                "Tower Guardian",
                description="A glowing spirit hovers beside the pedestal.",
                dialogue_topics=guardian_topics,
                riddle=guardian_riddle,
                surrender_reject_lines=[
                    "The trial cannot be escaped. Only answered or endured."
                ],
                inventory=["star_crystal"],
                defeat_lines=["The guardian dissolves into drifting sparks of light."],
                reward_flags={"guardian_passed"},
            ),
        )

    def scatter_items(self) -> None:
        self.place_item("forest", "moonleaf_herb")
        self.place_item("forest", "rope")
        self.place_item("forest", "iron_sword")
        self.place_item("forest", "health_tonic")

    def build_encounters(self) -> None:
        self.add_encounter_rule(
            EncounterRule(
                "guard_confrontation",
                locations={"village"},
                chance=1.0,
                blocked_flags={"jailed", "game_won"},
                predicate=lambda game: game.bounty > 0
                and self.guard_alive("Guard Halwen"),
                handler=self.handle_guard_confrontation,
            )
        )
        self.add_encounter_rule(
            EncounterRule(
                "forest_small_spider",
                locations={"forest"},
                chance=0.30,
                blocked_flags={"jailed"},
                predicate=lambda game: game.player.is_alive(),
                handler=self.handle_forest_spider_encounter,
            )
        )

    def guard_alive(self, name: str) -> bool:
        guard = next(
            (npc for npc in self.get("village").npcs if npc.name == name), None
        )
        return guard is not None and guard.is_alive() and not guard.defeated

    def handle_guard_confrontation(self, game: Game) -> bool:
        guard = next(
            (npc for npc in self.get("village").npcs if npc.name == "Guard Halwen"),
            None,
        )
        if guard is None or guard.defeated or not guard.is_alive():
            return False
        action_separator("A guard confronts you")
        say(
            guard.name,
            f"You have a bounty of {game.bounty} gold on your head. Stand down.",
        )
        guard.hostile = True
        game.enter_mode("combat", npc=guard)
        return True

    def handle_forest_spider_encounter(self, game: Game) -> bool:
        spider = NPC(
            name="Small Spider",
            base_stats=Stats(max_hp=10, attack_min=3, attack_max=5, defense=0),
            description="A skittering spider drops from a branch, startled by your movement.",
            tags={"monster", "encounter"},
            aggression=80,
            courage=20,
            willingness_to_trade=0,
            hostile=True,
            surrender_reject_lines=["The spider only hisses and darts forward."],
            defeat_lines=["The small spider curls up and goes still among the leaves."],
            reward_gold=8,
            persistent=False,
        )
        action_separator("A forest encounter")
        print("A small spider drops from the branches and rushes toward you!")
        game.enter_mode("combat", npc=spider)
        return True

    def on_item_picked_up(self, game: Game, item_id: str) -> None:
        if item_id == "star_crystal":
            game.flags.add("has_star_crystal")

    def on_npc_defeated(self, game: Game, npc: NPC) -> None:
        if npc.name == "Crystal Spider":
            print(
                "The crystal web tears apart, revealing a safer route through the cave."
            )

    def on_location_enter(self, game: Game, location: Location) -> None:
        if location.key == "tower_top" and "first_tower_visit" not in game.flags:
            game.flags.add("first_tower_visit")
            print(
                "The air grows brighter and heavier at the same time. Something ancient is waiting here."
            )


# ============================================================
# Engines
# ============================================================


class Engine:
    def run(self, game: Game) -> None:
        raise NotImplementedError


class ExplorationEngine(Engine):
    def run(self, game: Game) -> None:
        if game.world.try_start_of_exploration_encounter(game):
            return

        location = game.current_location()
        divider(f"{game.world.name} - {location.name}")
        print(location.description)
        print(
            f"\n[HP {game.player.health}/{game.player.total_max_hp(game.world.item_db)}] [Gold {game.player.gold}] [Bounty {game.bounty}]"
        )

        if location.items:
            print("\nItems here:")
            for item_id in location.items:
                print(f"- {game.item_name(item_id)}")

        if location.npcs:
            print("\nPeople / creatures here:")
            for npc in location.npcs:
                print(f"- {npc.name} ({npc.mood_label()})")

        if location.features:
            print("\nNotable features:")
            for feature in location.features:
                print(f"- {feature.name}")

        game.show_goal()

        choices = []
        jailed = "jailed" in game.flags and location.key == "jail"
        if not jailed:
            for exit_obj in location.exits:
                destination = game.world.get(exit_obj.destination)
                text = f"Go {exit_obj.direction} to {destination.name}"
                if game.exit_is_blocked(exit_obj):
                    text += " [blocked]"
                elif exit_obj.warning_text:
                    text += " [risky]"
                choices.append({"kind": "move", "text": text, "value": exit_obj})

            for item_id in location.items:
                choices.append(
                    {
                        "kind": "take_item",
                        "text": f"Take {game.item_name(item_id)}",
                        "value": item_id,
                    }
                )

            for npc in location.npcs:
                choices.append({"kind": "npc", "text": npc.menu_text(), "value": npc})

        for feature in location.features:
            choices.append(
                {"kind": "feature", "text": feature.menu_text(), "value": feature}
            )

        if not jailed:
            choices.append({"kind": "menu", "text": "Open menu", "value": None})

        choice = game.choose(choices, prompt="\nWhat do you want to do? ")

        if choice["kind"] == "move":
            game.handle_move(choice["value"])
        elif choice["kind"] == "take_item":
            game.handle_take_item(choice["value"])
        elif choice["kind"] == "npc":
            game.enter_mode("npc", npc=choice["value"])
        elif choice["kind"] == "feature":
            choice["value"].interact(game)
        elif choice["kind"] == "menu":
            game.enter_mode("menu")


class NPCInteractionEngine(Engine):
    def run(self, game: Game) -> None:
        npc: NPC = game.context["npc"]

        divider(f"Interacting with {npc.name}")
        print(npc.description)
        print(f"Mood: {npc.mood_label()}")
        print(f"Health: {npc.health}/{npc.total_max_hp(game.world.item_db)}")
        print(f"Gold: {npc.gold}")

        options = []
        if npc.can_talk(game):
            options.append(
                {"kind": "talk", "text": f"Talk to {npc.name}", "value": npc}
            )
        if npc.can_trade(game):
            options.append(
                {"kind": "trade", "text": f"Trade with {npc.name}", "value": npc}
            )
        if npc.can_riddle():
            options.append(
                {"kind": "riddle", "text": f"Challenge {npc.name}", "value": npc}
            )
        if npc.can_attack():
            options.append(
                {"kind": "attack", "text": f"Attack {npc.name}", "value": npc}
            )
        if npc.can_loot():
            options.append({"kind": "loot", "text": f"Loot {npc.name}", "value": npc})
        options.append({"kind": "back", "text": "Step away", "value": None})

        choice = game.choose(options, prompt="\nChoose an interaction: ")
        kind = choice["kind"]
        if kind == "talk":
            game.enter_mode("dialogue", npc=npc)
        elif kind == "trade":
            game.enter_mode("trade", npc=npc)
        elif kind == "riddle":
            game.enter_mode("riddle", npc=npc)
        elif kind == "attack":
            npc.on_player_attack(game)
            game.enter_mode("combat", npc=npc)
        elif kind == "loot":
            game.loot_npc(npc)
            game.enter_mode("npc", npc=npc)
        elif kind == "back":
            game.enter_mode("exploration")


class DialogueEngine(Engine):
    def run(self, game: Game) -> None:
        npc: NPC = game.context["npc"]
        topics = npc.available_topics(game)
        if not topics:
            say(npc.name, "I have nothing more to say right now.")
            game.enter_mode("npc", npc=npc)
            return

        divider(f"Dialogue - {npc.name}")
        options = [
            {"kind": "topic", "text": topic.title, "value": topic} for topic in topics
        ]
        options.append({"kind": "back", "text": "Back", "value": None})
        choice = game.choose(options, prompt="\nChoose a topic: ")
        if choice["kind"] == "back":
            game.enter_mode("npc", npc=npc)
            return

        topic: DialogueTopic = choice["value"]
        for line in topic.lines:
            say(npc.name, line)
        if topic.outcome_effect is not None:
            topic.outcome_effect.apply(game)
        if topic.once:
            npc.used_topics.add(topic.key)
        if "game_won" in game.flags:
            game.running = False
            return
        game.enter_mode("npc", npc=npc)


class TradeEngine(Engine):
    def run(self, game: Game) -> None:
        npc: NPC = game.context["npc"]
        offers = npc.available_trade_offers(game)
        divider(f"Trade - {npc.name}")
        print(f"Your gold: {game.player.gold}")
        if not offers:
            say(npc.name, "I have nothing I can trade right now.")
            game.enter_mode("npc", npc=npc)
            return

        options = []
        for offer_index, offer in offers:
            wants_parts = []
            if offer.wants_gold:
                wants_parts.append(f"{offer.wants_gold} gold")
            if offer.wants_items:
                wants_parts.extend(
                    game.item_name(item_id) for item_id in offer.wants_items
                )
            gives_parts = []
            if offer.gives_gold:
                gives_parts.append(f"{offer.gives_gold} gold")
            if offer.gives_items:
                gives_parts.extend(
                    game.item_name(item_id) for item_id in offer.gives_items
                )
            wants = ", ".join(wants_parts) or "nothing"
            gives = ", ".join(gives_parts) or "nothing"
            options.append(
                {
                    "kind": "offer",
                    "text": f"{offer.title} [{wants} -> {gives}]",
                    "value": (offer_index, offer),
                }
            )
        options.append({"kind": "back", "text": "Back", "value": None})

        choice = game.choose(options, prompt="\nChoose a trade: ")
        if choice["kind"] == "back":
            game.enter_mode("npc", npc=npc)
            return

        offer_index, offer = choice["value"]
        if game.player.gold < offer.wants_gold:
            say(npc.name, f"Come back when you have {offer.wants_gold} gold.")
            game.enter_mode("npc", npc=npc)
            return
        if not game.player.has_items(offer.wants_items):
            say(
                npc.name,
                f"Come back when you have: {', '.join(game.item_name(i) for i in offer.wants_items)}.",
            )
            game.enter_mode("npc", npc=npc)
            return
        if npc.gold < offer.gives_gold or not npc.has_items(offer.gives_items):
            say(npc.name, "I cannot complete that trade right now.")
            game.enter_mode("npc", npc=npc)
            return

        game.player.gold -= offer.wants_gold
        npc.gold += offer.wants_gold
        npc.gold -= offer.gives_gold
        game.player.gold += offer.gives_gold
        game.player.remove_items(offer.wants_items)
        npc.add_items(offer.wants_items)
        npc.remove_items(offer.gives_items)
        for item_id in offer.gives_items:
            game.give_player_item(item_id)
        if not offer.repeatable:
            npc.completed_trades.add(offer_index)
        say(npc.name, "A fair trade.")
        game.enter_mode("npc", npc=npc)


class RiddleEngine(Engine):
    def run(self, game: Game) -> None:
        npc: NPC = game.context["npc"]
        riddle = npc.riddle
        if riddle is None:
            game.enter_mode("npc", npc=npc)
            return

        divider(f"Challenge - {npc.name}")
        if npc.riddle_solved:
            for line in riddle.repeat_lines:
                say(npc.name, line)
            game.enter_mode("npc", npc=npc)
            return

        for line in riddle.intro_lines:
            say(npc.name, line)
        say(npc.name, riddle.question)
        answer = input("Your answer: ").strip().lower()

        if answer in [a.lower() for a in riddle.answers]:
            for line in riddle.success_lines:
                say(npc.name, line)
            npc.riddle_solved = True
            for flag in riddle.set_flags_on_success:
                game.flags.add(flag)
        else:
            for line in riddle.failure_lines:
                say(npc.name, line)
            if riddle.damage_on_failure > 0:
                actual = game.player.take_damage(
                    riddle.damage_on_failure, game.world.item_db
                )
                print(f"You take {actual} damage.")

        game.enter_mode("npc", npc=npc)


class CombatEngine(Engine):
    def run(self, game: Game) -> None:
        enemy: NPC = game.context["npc"]
        player_defending = False
        enemy_defending = False

        divider(f"Combat - {enemy.name}")
        print(f"{enemy.name} squares off against you.")

        while (
            game.running
            and game.player.is_alive()
            and enemy.is_alive()
            and enemy.hostile
        ):
            print("\n--- Combat Round ---")
            print(
                f"Your HP: {game.player.health}/{game.player.total_max_hp(game.world.item_db)}"
            )
            print(
                f"{enemy.name} HP: {enemy.health}/{enemy.total_max_hp(game.world.item_db)}"
            )
            print(f"Your gold: {game.player.gold} | Their gold: {enemy.gold}")

            action, payload = game.player.controller.choose_combat_action(
                game, game.player, enemy
            )
            if action == "none":
                continue

            if action == "attack":
                raw = game.player.roll_attack_damage(game.world.item_db)
                if enemy_defending:
                    raw = max(1, raw // 2)
                    enemy_defending = False
                actual = enemy.take_damage(raw, game.world.item_db)
                print(f"You hit {enemy.name} for {actual} damage.")

                if enemy.maybe_surrender(game):
                    if enemy.name == "Bandit Nox":
                        game.flags.add("bandit_spared")
                    if game.resolve_enemy_surrender(enemy):
                        return

                if not enemy.is_alive():
                    enemy.on_defeat(game)
                    if enemy.persistent:
                        game.enter_mode("npc", npc=enemy)
                    else:
                        game.enter_mode("exploration")
                    return

            elif action == "defend":
                player_defending = True
                print("You brace for the next attack.")

            elif action == "item":
                ok, message, healed = game.player.use_consumable(
                    payload, game.world.item_db
                )
                if ok:
                    print(f"{message} Restored {healed} HP.")
                else:
                    print(message)
                    continue

            elif action == "run":
                if random.random() < 0.5:
                    print("You escape successfully!")
                    enemy.hostile = False if enemy.surrendered else enemy.hostile
                    game.enter_mode("exploration")
                    return
                print("You try to run, but the enemy cuts you off!")

            elif action == "surrender":
                if enemy.handle_player_surrender(game):
                    return
                continue

            if enemy.is_alive() and enemy.hostile:
                enemy_action, enemy_payload = enemy.controller.choose_combat_action(
                    game, enemy, game.player
                )
                if enemy_action == "item" and enemy_payload is not None:
                    ok, msg, healed = enemy.use_consumable(
                        enemy_payload, game.world.item_db
                    )
                    if ok:
                        print(
                            f"{enemy.name} uses {game.item_name(enemy_payload)} and heals {healed} HP."
                        )
                elif enemy_action == "defend":
                    enemy_defending = True
                    print(f"{enemy.name} takes a guarded stance.")
                else:
                    raw = enemy.roll_attack_damage(game.world.item_db)
                    if player_defending:
                        raw = max(1, raw // 2)
                        player_defending = False
                    actual = game.player.take_damage(raw, game.world.item_db)
                    print(f"{enemy.name} hits you for {actual} damage.")
                if not game.player.is_alive():
                    return

        if enemy.persistent:
            game.enter_mode("npc", npc=enemy)
        else:
            game.enter_mode("exploration")


class MenuEngine(Engine):
    def run(self, game: Game) -> None:
        divider("MENU")
        options = [
            {"kind": "status", "text": "View status", "value": None},
            {"kind": "loadout", "text": "Manage loadout", "value": None},
            {"kind": "journal", "text": "View world flags / journal", "value": None},
            {"kind": "back", "text": "Return to exploration", "value": None},
            {"kind": "quit", "text": "Quit game", "value": None},
        ]
        choice = game.choose(options, prompt="Choose a menu option: ")
        if choice["kind"] == "status":
            game.player.show_status(game.world.item_db, bounty=game.bounty)
            game.enter_mode("menu")
        elif choice["kind"] == "loadout":
            game.enter_mode("loadout")
        elif choice["kind"] == "journal":
            game.show_journal()
            game.enter_mode("menu")
        elif choice["kind"] == "back":
            game.enter_mode("exploration")
        elif choice["kind"] == "quit":
            print("Thanks for playing!")
            game.running = False


class LoadoutEngine(Engine):
    def run(self, game: Game) -> None:
        divider("LOADOUT")
        player = game.player
        item_db = game.world.item_db

        print("Current equipment:")
        for slot in player.EQUIPMENT_SLOTS:
            item_id = player.equipment[slot]
            label = game.item_name(item_id) if item_id is not None else "empty"
            print(f"- {slot}: {label}")

        options = []
        for item_id in player.equippable_items(item_db):
            item = item_db[item_id]
            options.append(
                {
                    "kind": "equip",
                    "text": f"Equip {item.name} ({item.slot})",
                    "value": item_id,
                }
            )
        for slot in player.EQUIPMENT_SLOTS:
            if player.equipment[slot] is not None:
                options.append(
                    {"kind": "unequip", "text": f"Unequip {slot}", "value": slot}
                )
        options.append({"kind": "back", "text": "Back", "value": None})

        choice = game.choose(options, prompt="Choose a loadout action: ")
        if choice["kind"] == "equip":
            ok, message = player.equip(choice["value"], item_db)
            print(message)
            game.enter_mode("loadout")
        elif choice["kind"] == "unequip":
            ok, message = player.unequip(choice["value"], item_db)
            print(message)
            game.enter_mode("loadout")
        elif choice["kind"] == "back":
            game.enter_mode("menu")


# ============================================================
# Game director / router
# ============================================================


class Game:
    def __init__(self, player_name: str = "Tav", world: World | None = None):
        self.world = world or StarCrystalWorld()
        self.player = Character(
            name=player_name,
            base_stats=Stats(max_hp=100, attack_min=6, attack_max=10, defense=1),
            controller=PlayerController(),
            location="village",
            gold=5,
        )
        self.flags: set[str] = {"game_started"}
        self.bounty = 0
        self.running = True
        self.mode = "exploration"
        self.context: dict[str, object] = {}
        self.engines: dict[str, Engine] = {
            "exploration": ExplorationEngine(),
            "npc": NPCInteractionEngine(),
            "dialogue": DialogueEngine(),
            "trade": TradeEngine(),
            "riddle": RiddleEngine(),
            "combat": CombatEngine(),
            "menu": MenuEngine(),
            "loadout": LoadoutEngine(),
        }
        self.world.on_location_enter(self, self.current_location())

    def item_name(self, item_id: str | None) -> str:
        if item_id is None:
            return "empty"
        return self.world.item(item_id).name

    def current_location(self) -> Location:
        return self.world.get(self.player.location)

    def find_npc(self, npc_name: str) -> NPC | None:
        for location in self.world.locations.values():
            for npc in location.npcs:
                if npc.name == npc_name:
                    return npc
        return None

    def enter_mode(self, mode: str, **context: object) -> None:
        self.mode = mode
        self.context = context

    def change_bounty(self, amount: int, reason: str | None = None) -> None:
        self.bounty = max(0, self.bounty + amount)
        if reason:
            print(f"[Bounty] {reason}: {self.bounty}")
        else:
            print(f"[Bounty] {self.bounty}")

    def show_goal(self) -> None:
        if "jailed" in self.flags:
            print("\nGoal: Serve your time.")
        elif "quest_started" not in self.flags:
            print("\nGoal: Talk to Elder Mira.")
        elif "has_star_crystal" in self.flags and "game_won" not in self.flags:
            print("\nGoal: Return the Star Crystal to Elder Mira.")
        elif "game_won" in self.flags:
            print("\nGoal: The valley has been saved.")
        else:
            print("\nGoal: Explore the valley and recover the Star Crystal.")

    def show_journal(self) -> None:
        divider("JOURNAL / WORLD FLAGS")
        print(f"Gold: {self.player.gold}")
        print(f"Bounty: {self.bounty}")
        if not self.flags:
            print("No flags set.")
            return
        for flag in sorted(self.flags):
            print(f"- {flag}")

    def choose(self, options: list[dict], prompt: str = "Choose: ") -> dict:
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option['text']}")
        while True:
            answer = input(prompt).strip()
            try:
                idx = int(answer) - 1
                selected = options[idx]
                action_separator(selected.get("text"))
                return selected
            except (ValueError, IndexError):
                print("Please enter a valid menu number.")

    def exit_is_blocked(self, exit_obj: Exit) -> bool:
        if exit_obj.blocked:
            return True
        if exit_obj.requires_item and not self.player.has_item(exit_obj.requires_item):
            return True
        if exit_obj.requires_flag and exit_obj.requires_flag not in self.flags:
            return True
        return False

    def handle_move(self, exit_obj: Exit) -> None:
        if exit_obj.on_attempt_effect is not None:
            exit_obj.on_attempt_effect.apply(self)

        if self.exit_is_blocked(exit_obj):
            print(exit_obj.blocked_text)
            if exit_obj.on_blocked_effect is not None:
                exit_obj.on_blocked_effect.apply(self)
            return

        if exit_obj.warning_text:
            print(exit_obj.warning_text)
            confirm = input("Continue? [y/N] ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("You decide not to risk it.")
                return

        self.player.location = exit_obj.destination
        self.flags.add(f"visited:{exit_obj.destination}")
        if exit_obj.on_success_effect is not None:
            exit_obj.on_success_effect.apply(self)
        self.world.on_location_enter(self, self.current_location())

    def give_player_item(self, item_id: str) -> None:
        self.player.add_item(item_id)
        item = self.world.item(item_id)
        print(f"[Inventory] You got: {item.name}")
        if item.description:
            print(item.description)
        for flag in item.set_flags_on_pickup:
            self.flags.add(flag)
        self.world.on_item_picked_up(self, item_id)

    def handle_take_item(self, item_id: str) -> None:
        location = self.current_location()
        if item_id not in location.items:
            print("That item is no longer here.")
            return
        location.items.remove(item_id)
        self.give_player_item(item_id)

    def loot_npc(self, npc: NPC) -> None:
        if not npc.can_loot():
            print(f"There is nothing to loot from {npc.name}.")
            return
        divider(f"Looting {npc.name}")
        if npc.gold > 0:
            print(f"You take {npc.gold} gold.")
            self.player.gold += npc.gold
            npc.gold = 0
        while npc.inventory:
            seen = set()
            options = []
            for item_id in npc.inventory:
                if item_id in seen:
                    continue
                seen.add(item_id)
                count = npc.inventory.count(item_id)
                suffix = f" x{count}" if count > 1 else ""
                options.append(
                    {
                        "kind": "take",
                        "text": f"Take {self.item_name(item_id)}{suffix}",
                        "value": item_id,
                    }
                )
            options.append({"kind": "done", "text": "Done looting", "value": None})
            choice = self.choose(options, prompt="Choose loot: ")
            if choice["kind"] == "done":
                return
            item_id = choice["value"]
            npc.remove_item(item_id)
            self.give_player_item(item_id)
        print("Nothing remains.")

    def resolve_enemy_surrender(self, enemy: NPC) -> bool:
        divider(f"{enemy.name} Surrenders")
        options = [
            {"kind": "spare", "text": f"Spare {enemy.name}", "value": None},
            {"kind": "kill", "text": f"Kill {enemy.name}", "value": None},
        ]
        choice = self.choose(options, prompt="Choose your response: ")
        if choice["kind"] == "spare":
            if enemy.name == "Bandit Nox":
                self.flags.add("bandit_spared")
            self.enter_mode("npc", npc=enemy)
            return True
        if choice["kind"] == "kill":
            enemy.health = 0
            enemy.on_defeat(self)
            if enemy.persistent:
                self.enter_mode("npc", npc=enemy)
            else:
                self.enter_mode("exploration")
            return True
        return False

    def run(self) -> None:
        divider(f"Welcome to {self.world.name}")
        print(
            "This prototype uses Character + Controller, effect-driven paths and features, gold-based trade, and encounter rules that can fire as you explore."
        )
        while self.running:
            if not self.player.is_alive():
                divider("Defeat")
                print("You collapse from your injuries. The adventure ends here.")
                break
            engine = self.engines[self.mode]
            engine.run(self)
            if self.running and self.player.is_alive():
                prompt_continue()


# ============================================================
# Textual front-end (v5 architecture, fused inline)
# ============================================================
# Same structure as v6: QueueWriter + BackendBridge + RPGTextualApp.
# In this version the snapshots and log lines may also include ASCII
# art keys so the UI can render the matching illustration.

"""Textual front end for rich_single_file_rpg.py.

Development version that imports the core game from a sibling file.
A build script can merge both files into a single-file program by setting
EMBEDDED_CORE=True and prepending the core code.
"""

from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import builtins
import importlib.util
from pathlib import Path
import queue
import sys
import threading
import traceback
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static


EMBEDDED_CORE = True
CORE_PATH = Path(__file__).with_name("rich_single_file_rpg.py")


def load_core_module() -> Any:
    if EMBEDDED_CORE:
        return sys.modules[__name__]
    spec = importlib.util.spec_from_file_location("rpg_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load game core from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QueueWriter:
    """Capture stdout/stderr and emit complete lines to the UI queue."""

    def __init__(self, bridge: "BackendBridge"):
        self.bridge = bridge
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.bridge.emit("log", text=line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self.bridge.emit("log", text=self._buffer)
            self._buffer = ""


class BackendBridge:
    def __init__(self, core: Any):
        self.core = core
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.responses: queue.Queue[Any] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.game: Any | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run_backend, daemon=True)
        self.thread.start()

    def emit(self, event_type: str, **payload: Any) -> None:
        self.events.put({"type": event_type, **payload})

    def snapshot(self) -> dict[str, Any]:
        game = self.game
        if game is None:
            return {
                "mode": "boot",
                "goal": "",
                "location": "Starting...",
                "description": "",
                "items": [],
                "npcs": [],
                "features": [],
                "health": 0,
                "max_health": 0,
                "gold": 0,
                "bounty": 0,
                "attack": (0, 0),
                "defense": 0,
                "equipment": {},
                "inventory": [],
                "flags": [],
            }
        location = game.current_location()
        item_db = game.world.item_db
        low, high = game.player.attack_range(item_db)
        counts = Counter(game.player.inventory)
        inventory = []
        seen = set()
        for item_id in game.player.inventory:
            if item_id in seen:
                continue
            seen.add(item_id)
            item = item_db[item_id]
            count = counts[item_id]
            inventory.append((item.name, count))
        equipment = {}
        for slot in game.player.EQUIPMENT_SLOTS:
            item_id = game.player.equipment[slot]
            equipment[slot] = item_db[item_id].name if item_id is not None else "empty"
        focus_npc = game.context.get("npc") if isinstance(game.context, dict) else None
        focus_npc_name = None
        focus_npc_state = None
        focus_npc_mood = None
        if focus_npc is not None and hasattr(focus_npc, "name"):
            focus_npc_name = focus_npc.name
            if getattr(focus_npc, "defeated", False):
                focus_npc_state = "dead"
            elif getattr(focus_npc, "is_alive", lambda: False)():
                focus_npc_state = "alive"
            else:
                focus_npc_state = "unknown"
            if hasattr(focus_npc, "mood_label"):
                focus_npc_mood = focus_npc.mood_label()
        return {
            "mode": game.mode,
            "goal": self.goal_text(game),
            "location_key": location.key,
            "location": location.name,
            "description": location.description,
            "items": [game.item_name(item_id) for item_id in location.items],
            "npcs": [f"{npc.name} ({npc.mood_label()})" for npc in location.npcs],
            "features": [feature.name for feature in location.features],
            "health": game.player.health,
            "max_health": game.player.total_max_hp(item_db),
            "gold": game.player.gold,
            "bounty": game.bounty,
            "attack": (low, high),
            "defense": game.player.total_defense(item_db),
            "equipment": equipment,
            "inventory": inventory,
            "flags": sorted(game.flags),
            "focus_npc_name": focus_npc_name,
            "focus_npc_state": focus_npc_state,
            "focus_npc_mood": focus_npc_mood,
        }

    def goal_text(self, game: Any) -> str:
        if "jailed" in game.flags:
            return "Serve your time."
        if "quest_started" not in game.flags:
            return "Talk to Elder Mira."
        if "has_star_crystal" in game.flags and "game_won" not in game.flags:
            return "Return the Star Crystal to Elder Mira."
        if "game_won" in game.flags:
            return "The valley has been saved."
        return "Explore the valley and recover the Star Crystal."

    def request_choice(self, options: list[dict[str, Any]], prompt: str) -> int:
        self.emit("state", snapshot=self.snapshot())
        self.emit(
            "choices", prompt=prompt, options=[option["text"] for option in options]
        )
        return int(self.responses.get())

    def request_text(self, prompt: str) -> str:
        self.emit("state", snapshot=self.snapshot())
        self.emit("text_input", prompt=prompt)
        response = self.responses.get()
        return "" if response is None else str(response)

    def request_continue(self, prompt: str) -> None:
        # In the TUI, the current outcome remains visible in its own pane, so
        # an explicit continue step is unnecessary. Treat it as a no-op.
        self.emit("state", snapshot=self.snapshot())
        return

    def _run_backend(self) -> None:
        core = self.core
        bridge = self

        class TuiExplorationEngine(core.Engine):
            def run(self, game: Any) -> None:
                if game.world.try_start_of_exploration_encounter(game):
                    return

                location = game.current_location()
                choices = []
                jailed = "jailed" in game.flags and location.key == "jail"
                if not jailed:
                    for exit_obj in location.exits:
                        destination = game.world.get(exit_obj.destination)
                        text = f"Go {exit_obj.direction} to {destination.name}"
                        if game.exit_is_blocked(exit_obj):
                            text += " [blocked]"
                        elif exit_obj.warning_text:
                            text += " [risky]"
                        choices.append(
                            {"kind": "move", "text": text, "value": exit_obj}
                        )
                    for item_id in location.items:
                        choices.append(
                            {
                                "kind": "take_item",
                                "text": f"Take {game.item_name(item_id)}",
                                "value": item_id,
                            }
                        )
                    for npc in location.npcs:
                        choices.append(
                            {"kind": "npc", "text": npc.menu_text(), "value": npc}
                        )
                for feature in location.features:
                    choices.append(
                        {
                            "kind": "feature",
                            "text": feature.menu_text(),
                            "value": feature,
                        }
                    )
                if not jailed:
                    choices.append({"kind": "menu", "text": "Open menu", "value": None})

                choice = game.choose(choices, prompt="What do you want to do?")
                if choice["kind"] == "move":
                    game.handle_move(choice["value"])
                elif choice["kind"] == "take_item":
                    game.handle_take_item(choice["value"])
                elif choice["kind"] == "npc":
                    game.enter_mode("npc", npc=choice["value"])
                elif choice["kind"] == "feature":
                    choice["value"].interact(game)
                elif choice["kind"] == "menu":
                    game.enter_mode("menu")

        class TuiNPCInteractionEngine(core.Engine):
            def run(self, game: Any) -> None:
                npc = game.context["npc"]
                if (
                    npc.is_guard()
                    and game.bounty > 0
                    and npc.is_alive()
                    and not npc.defeated
                    and not npc.hostile
                ):
                    core.say(
                        npc.name, f"You have a bounty of {game.bounty}. Stand down."
                    )
                    npc.hostile = True
                    game.enter_mode("combat", npc=npc)
                    return

                options = []
                if npc.can_talk(game):
                    options.append(
                        {"kind": "talk", "text": f"Talk to {npc.name}", "value": npc}
                    )
                if npc.can_trade(game):
                    options.append(
                        {
                            "kind": "trade",
                            "text": f"Trade with {npc.name}",
                            "value": npc,
                        }
                    )
                if npc.can_riddle():
                    options.append(
                        {
                            "kind": "riddle",
                            "text": f"Challenge {npc.name}",
                            "value": npc,
                        }
                    )
                if npc.can_attack():
                    options.append(
                        {"kind": "attack", "text": f"Attack {npc.name}", "value": npc}
                    )
                if npc.can_loot():
                    options.append(
                        {"kind": "loot", "text": f"Loot {npc.name}", "value": npc}
                    )
                options.append({"kind": "back", "text": "Step away", "value": None})

                choice = game.choose(
                    options, prompt=f"How do you want to interact with {npc.name}?"
                )
                kind = choice["kind"]
                if kind == "talk":
                    game.enter_mode("dialogue", npc=npc)
                elif kind == "trade":
                    game.enter_mode("trade", npc=npc)
                elif kind == "riddle":
                    game.enter_mode("riddle", npc=npc)
                elif kind == "attack":
                    npc.on_player_attack(game)
                    game.enter_mode("combat", npc=npc)
                elif kind == "loot":
                    game.loot_npc(npc)
                    game.enter_mode("npc", npc=npc)
                elif kind == "back":
                    game.enter_mode("exploration")

        class TuiDialogueEngine(core.Engine):
            def run(self, game: Any) -> None:
                npc = game.context["npc"]
                topics = npc.available_topics(game)
                if not topics:
                    core.say(npc.name, "I have nothing more to say right now.")
                    game.enter_mode("npc", npc=npc)
                    return

                options = [
                    {"kind": "topic", "text": topic.title, "value": topic}
                    for topic in topics
                ]
                options.append({"kind": "back", "text": "Back", "value": None})
                choice = game.choose(
                    options, prompt=f"What do you want to ask {npc.name}?"
                )
                if choice["kind"] == "back":
                    game.enter_mode("npc", npc=npc)
                    return

                topic = choice["value"]
                for line in topic.lines:
                    core.say(npc.name, line)
                if topic.outcome_effect is not None:
                    topic.outcome_effect.apply(game)
                if topic.once:
                    npc.used_topics.add(topic.key)
                if "game_won" in game.flags:
                    game.running = False
                    return
                game.enter_mode("npc", npc=npc)

        class TuiTradeEngine(core.Engine):
            def run(self, game: Any) -> None:
                npc = game.context["npc"]
                offers = npc.available_trade_offers(game)
                if not offers:
                    core.say(npc.name, "I have nothing I can trade right now.")
                    game.enter_mode("npc", npc=npc)
                    return

                options = []
                for offer_index, offer in offers:
                    wants_parts = []
                    if offer.wants_gold:
                        wants_parts.append(f"{offer.wants_gold} gold")
                    if offer.wants_items:
                        wants_parts.extend(
                            game.item_name(item_id) for item_id in offer.wants_items
                        )
                    gives_parts = []
                    if offer.gives_gold:
                        gives_parts.append(f"{offer.gives_gold} gold")
                    if offer.gives_items:
                        gives_parts.extend(
                            game.item_name(item_id) for item_id in offer.gives_items
                        )
                    wants = ", ".join(wants_parts) or "nothing"
                    gives = ", ".join(gives_parts) or "nothing"
                    options.append(
                        {
                            "kind": "offer",
                            "text": f"{offer.title} [{wants} -> {gives}]",
                            "value": (offer_index, offer),
                        }
                    )
                options.append({"kind": "back", "text": "Back", "value": None})

                choice = game.choose(options, prompt=f"Choose a trade with {npc.name}")
                if choice["kind"] == "back":
                    game.enter_mode("npc", npc=npc)
                    return

                offer_index, offer = choice["value"]
                if game.player.gold < offer.wants_gold:
                    core.say(
                        npc.name, f"Come back when you have {offer.wants_gold} gold."
                    )
                    game.enter_mode("npc", npc=npc)
                    return
                if not game.player.has_items(offer.wants_items):
                    core.say(
                        npc.name,
                        f"Come back when you have: {', '.join(game.item_name(i) for i in offer.wants_items)}.",
                    )
                    game.enter_mode("npc", npc=npc)
                    return
                if npc.gold < offer.gives_gold or not npc.has_items(offer.gives_items):
                    core.say(npc.name, "I cannot complete that trade right now.")
                    game.enter_mode("npc", npc=npc)
                    return

                game.player.gold -= offer.wants_gold
                npc.gold += offer.wants_gold
                npc.gold -= offer.gives_gold
                game.player.gold += offer.gives_gold
                game.player.remove_items(offer.wants_items)
                npc.add_items(offer.wants_items)
                npc.remove_items(offer.gives_items)
                for item_id in offer.gives_items:
                    game.give_player_item(item_id)
                if not offer.repeatable:
                    npc.completed_trades.add(offer_index)
                core.say(npc.name, "A fair trade.")
                game.enter_mode("npc", npc=npc)

        class TuiRiddleEngine(core.Engine):
            def run(self, game: Any) -> None:
                npc = game.context["npc"]
                riddle = npc.riddle
                if riddle is None:
                    game.enter_mode("npc", npc=npc)
                    return
                if npc.riddle_solved:
                    for line in riddle.repeat_lines:
                        core.say(npc.name, line)
                    game.enter_mode("npc", npc=npc)
                    return

                for line in riddle.intro_lines:
                    core.say(npc.name, line)
                core.say(npc.name, riddle.question)
                answer = input("Your answer: ").strip().lower()
                if answer in [a.lower() for a in riddle.answers]:
                    for line in riddle.success_lines:
                        core.say(npc.name, line)
                    npc.riddle_solved = True
                    for flag in riddle.set_flags_on_success:
                        game.flags.add(flag)
                else:
                    for line in riddle.failure_lines:
                        core.say(npc.name, line)
                    if riddle.damage_on_failure > 0:
                        actual = game.player.take_damage(
                            riddle.damage_on_failure, game.world.item_db
                        )
                        print(f"You take {actual} damage.")
                game.enter_mode("npc", npc=npc)

        class TuiMenuEngine(core.Engine):
            def run(self, game: Any) -> None:
                options = [
                    {"kind": "status", "text": "View status", "value": None},
                    {"kind": "loadout", "text": "Manage loadout", "value": None},
                    {
                        "kind": "journal",
                        "text": "View world flags / journal",
                        "value": None,
                    },
                    {"kind": "back", "text": "Return to exploration", "value": None},
                    {"kind": "quit", "text": "Quit game", "value": None},
                ]
                choice = game.choose(options, prompt="Choose a menu option")
                if choice["kind"] == "status":
                    game.player.show_status(game.world.item_db, bounty=game.bounty)
                    game.enter_mode("menu")
                elif choice["kind"] == "loadout":
                    game.enter_mode("loadout")
                elif choice["kind"] == "journal":
                    game.show_journal()
                    game.enter_mode("menu")
                elif choice["kind"] == "back":
                    game.enter_mode("exploration")
                elif choice["kind"] == "quit":
                    print("Thanks for playing!")
                    game.running = False

        class TuiLoadoutEngine(core.Engine):
            def run(self, game: Any) -> None:
                player = game.player
                item_db = game.world.item_db
                options = []
                for item_id in player.equippable_items(item_db):
                    item = item_db[item_id]
                    current = player.equipment.get(item.slot)
                    suffix = " [equipped]" if current == item_id else ""
                    options.append(
                        {
                            "kind": "equip",
                            "text": f"Equip {item.name} ({item.slot}){suffix}",
                            "value": item_id,
                        }
                    )
                for slot in player.EQUIPMENT_SLOTS:
                    if player.equipment[slot] is not None:
                        options.append(
                            {
                                "kind": "unequip",
                                "text": f"Unequip {slot}",
                                "value": slot,
                            }
                        )
                options.append({"kind": "back", "text": "Back", "value": None})
                choice = game.choose(options, prompt="Choose a loadout action")
                if choice["kind"] == "equip":
                    ok, message = player.equip(choice["value"], item_db)
                    print(message)
                    game.enter_mode("loadout")
                elif choice["kind"] == "unequip":
                    ok, message = player.unequip(choice["value"], item_db)
                    print(message)
                    game.enter_mode("loadout")
                elif choice["kind"] == "back":
                    game.enter_mode("menu")

        class TuiGame(core.Game):
            def __init__(self, bridge: BackendBridge):
                self._bridge = bridge
                super().__init__(player_name="Tav", world=core.StarCrystalWorld())
                self.engines.update(
                    {
                        "exploration": TuiExplorationEngine(),
                        "npc": TuiNPCInteractionEngine(),
                        "dialogue": TuiDialogueEngine(),
                        "trade": TuiTradeEngine(),
                        "riddle": TuiRiddleEngine(),
                        "menu": TuiMenuEngine(),
                        "loadout": TuiLoadoutEngine(),
                    }
                )

            def choose(self, options: list[dict], prompt: str = "Choose: ") -> dict:
                index = self._bridge.request_choice(options, prompt)
                selected = options[index]
                core.action_separator(selected.get("text"))
                return selected

        old_input = builtins.input
        old_typewriter = core.typewriter_print
        old_continue = core.prompt_continue

        def queued_typewriter(prefix: str, text: str, word_delay: float = 0.02) -> None:
            bridge.emit("stream_start", prefix=prefix)
            words = text.split()
            for index, word in enumerate(words):
                suffix = " " if index < len(words) - 1 else ""
                bridge.emit("stream_chunk", text=word + suffix)
                if word_delay > 0:
                    time.sleep(word_delay)
            bridge.emit("stream_end")

        def queued_input(prompt: str = "") -> str:
            return bridge.request_text(prompt)

        def queued_continue(prompt: str = "Press Enter to continue...") -> None:
            bridge.request_continue(prompt)

        writer = QueueWriter(bridge)

        try:
            builtins.input = queued_input
            core.typewriter_print = queued_typewriter
            core.prompt_continue = queued_continue
            with redirect_stdout(writer), redirect_stderr(writer):
                bridge.game = TuiGame(bridge)
                bridge.emit("state", snapshot=bridge.snapshot())
                bridge.game.run()
            writer.flush()
            bridge.emit("state", snapshot=bridge.snapshot())
            bridge.emit("game_over", message="Game over.")
        except Exception:
            writer.flush()
            bridge.emit("log", text=traceback.format_exc())
            bridge.emit("game_over", message="The backend crashed.")
        finally:
            builtins.input = old_input
            core.typewriter_print = old_typewriter
            core.prompt_continue = old_continue


class RPGTextualApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
        layout: horizontal;
    }

    #sidebar {
        width: 34;
        min-width: 28;
        border: round $accent;
        padding: 1;
    }

    #main {
        width: 1fr;
        border: round $accent;
        padding: 1;
    }

    #actions {
        width: 40;
        min-width: 34;
        border: round $accent;
        padding: 1;
    }

    #goal, #location, #status {
        margin-bottom: 1;
        border: round $surface;
        padding: 1;
    }

    #ascii_art {
        height: 14;
        min-height: 10;
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
        content-align: center middle;
    }

    #current_event {
        height: 1fr;
        min-height: 8;
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
        overflow-y: auto;
    }

    #log_status {
        height: auto;
        margin-bottom: 1;
        border: round $surface;
        padding: 0 1;
    }

    #history {
        height: 6;
        min-height: 4;
        max-height: 12;
        border: round $surface;
        padding: 1;
    }

    #prompt {
        margin-bottom: 1;
        min-height: 3;
        border: round $surface;
        padding: 1;
    }

    #options {
        height: 1fr;
        margin-bottom: 1;
    }

    #options:focus {
        border: round $accent;
    }

    #text_input {
        margin-top: 1;
    }


    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("m", "menu", "Menu"),
        ("c", "clear_history", "Clear history"),
        ("l", "toggle_log", "Toggle log"),
        ("L", "toggle_log", "Toggle log"),
        ("escape", "focus_choices", "Focus choices"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.core = load_core_module()
        self.bridge = BackendBridge(self.core)
        self.interaction_mode = "boot"
        self.pending_options: list[str] = []
        self.current_event_text = ""
        self.game_over = False
        self.log_collapsed = True
        self.history_line_count = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("Goal", id="goal")
                yield Static("Location", id="location")
                yield Static("Status", id="status")
            with Vertical(id="main"):
                yield Static(self.make_ascii_art_placeholder({}), id="ascii_art")
                yield Static("Latest outcome will appear here.", id="current_event")
                yield Static("System log hidden. Press L to show.", id="log_status")
                yield RichLog(
                    id="history",
                    wrap=True,
                    markup=False,
                    auto_scroll=True,
                    max_lines=600,
                    classes="hidden",
                )
            with Vertical(id="actions"):
                yield Static("Starting up...", id="prompt")
                yield OptionList(id="options")
                yield Input(
                    placeholder="Type response and press Enter",
                    id="text_input",
                    classes="hidden",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.bridge.start()
        self.set_interval(0.05, self.poll_backend)

    def action_menu(self) -> None:
        if self.interaction_mode == "choice":
            for i, label in enumerate(self.pending_options):
                if label.lower().startswith("open menu"):
                    self.archive_current_event()
                    self.bridge.responses.put(i)
                    return

    def action_focus_choices(self) -> None:
        if self.interaction_mode == "choice":
            self.query_one("#options", OptionList).focus()
        elif self.interaction_mode == "text":
            self.query_one("#text_input", Input).focus()

    def action_clear_history(self) -> None:
        self.query_one("#history", RichLog).clear()
        self.history_line_count = 0
        self.update_history_size()

    def action_toggle_log(self) -> None:
        self.log_collapsed = not self.log_collapsed
        log = self.query_one("#history", RichLog)
        status = self.query_one("#log_status", Static)
        if self.log_collapsed:
            log.add_class("hidden")
            status.update("System log hidden. Press L to show.")
        else:
            log.remove_class("hidden")
            status.update("System log visible. Press L to hide.")
            self.update_history_size()

    def poll_backend(self) -> None:
        while True:
            try:
                event = self.bridge.events.get_nowait()
            except queue.Empty:
                break
            self.handle_backend_event(event)

    def handle_backend_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        if event_type == "log":
            self.append_current_event_line(event.get("text", ""))
        elif event_type == "stream_start":
            self.prepare_for_stream_start()
            self.append_current_event_chunk(event.get("prefix", ""))
        elif event_type == "stream_chunk":
            self.append_current_event_chunk(event.get("text", ""))
        elif event_type == "stream_end":
            self.append_current_event_chunk("\n")
        elif event_type == "state":
            self.update_state(event["snapshot"])
        elif event_type == "choices":
            self.show_choices(event["prompt"], event["options"])
        elif event_type == "text_input":
            self.show_text_input(event["prompt"])
        elif event_type == "game_over":
            self.game_over = True
            self.query_one("#prompt", Static).update(event.get("message", "Game over."))
            self.clear_options()
            self.hide_text_input()
            self.archive_current_event()

    def append_current_event_line(self, text: str) -> None:
        if self.current_event_text and not self.current_event_text.endswith("\n"):
            self.current_event_text += "\n"
        self.current_event_text += text
        self.render_current_event()

    def prepare_for_stream_start(self) -> None:
        if not self.current_event_text:
            return
        if not self.current_event_text.endswith("\n"):
            self.current_event_text += "\n"
        if not self.current_event_text.endswith("\n\n"):
            self.current_event_text += "\n"
        self.render_current_event()

    def append_current_event_chunk(self, text: str) -> None:
        self.current_event_text += text
        self.render_current_event()

    def normalize_event_text(self) -> str:
        if not self.current_event_text:
            return "Latest outcome will appear here."
        lines = self.current_event_text.split("\n")
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        if not lines:
            return "Latest outcome will appear here."
        normalized = []
        blank_run = 0
        for line in lines:
            if line == "":
                blank_run += 1
                if blank_run <= 1:
                    normalized.append(line)
            else:
                blank_run = 0
                normalized.append(line)
        return "\n".join(normalized)

    def render_current_event(self) -> None:
        self.query_one("#current_event", Static).update(self.normalize_event_text())

    def archive_current_event(self) -> None:
        text = self.normalize_event_text()
        if text and text != "Latest outcome will appear here.":
            history = self.query_one("#history", RichLog)
            history.write("─" * 56)
            self.history_line_count += 1
            for line in text.splitlines():
                history.write(line)
                self.history_line_count += 1
            self.update_history_size()
        self.current_event_text = ""
        self.render_current_event()

    def update_history_size(self) -> None:
        history = self.query_one("#history", RichLog)
        if self.log_collapsed:
            return
        desired = max(4, min(12, self.history_line_count + 2))
        history.styles.height = desired

    def choose_ascii_art_key(self, snapshot: dict[str, Any]) -> str:
        flags = set(snapshot.get("flags", []))
        location = snapshot.get("location", "")
        location_key = snapshot.get("location_key", "")
        latest = self.normalize_event_text().lower()
        focus_npc_name = snapshot.get("focus_npc_name")
        focus_npc_state = snapshot.get("focus_npc_state")

        if focus_npc_name:
            slug = focus_npc_name.lower().replace(" ", "_")
            return f"npc::{slug}::{focus_npc_state or 'alive'}"

        if "game_won" in flags:
            return "fountain_restored"
        if (
            "return the star crystal" in latest
            or "you found it... the star crystal" in latest
        ):
            return "return_crystal"
        if "surrenders" in latest or "spare" in latest or "make a deal" in latest:
            return "bandit_surrenders"
        if "shatters" in latest or "shatters on the rocks" in latest:
            return "item_shatter"
        if (
            "marches you to the village jail" in latest
            or "bounty" in latest
            and "stand down" in latest
        ):
            return "guard_arrest"
        if "the trial is complete" in latest or "wisdom and patience" in latest:
            return "guardian_trial_passed"
        if "not quite. think more carefully." in latest:
            return "guardian_trial_failed"
        if (
            "first_tower_visit" in flags
            or location_key == "tower_top"
            or location == "Tower Summit"
        ):
            return "first_tower_visit"

        location_icons = {
            "village": "loc::village",
            "crossroads": "loc::crossroads",
            "forest": "loc::forest",
            "garden": "loc::garden",
            "lake": "loc::lake",
            "cave_entrance": "loc::cave_entrance",
            "cave_depths": "loc::cave_depths",
            "ruins": "loc::ruins",
            "tower_gate": "loc::tower_gate",
            "tower_top": "loc::tower_top",
            "jail": "loc::jail",
        }
        if location_key in location_icons:
            return location_icons[location_key]

        # Backward-compatible fallback if the snapshot was created before location_key existed.
        legacy_location_icons = {
            "Sunmeadow Village": "loc::village",
            "Old Crossroads": "loc::crossroads",
            "Whispering Forest": "loc::forest",
            "Hidden Garden": "loc::garden",
            "Mirror Lake": "loc::lake",
            "Cave Entrance": "loc::cave_entrance",
            "Crystal Cave": "loc::cave_depths",
            "Moonstone Ruins": "loc::ruins",
            "Tower Gate": "loc::tower_gate",
            "Tower Summit": "loc::tower_top",
            "Village Jail": "loc::jail",
        }
        return legacy_location_icons.get(location, "loc::village")

    def make_npc_ascii_placeholder(
        self, name: str, state: str, mood: str | None = None
    ) -> str:
        mood_text = mood or "unknown"
        if state == "dead":
            art = "\n".join(
                [
                    "   .-------.",
                    "  /  x   x  \\",
                    " |     -     |",
                    r" |   \\___//  |",
                    r"  \\_________/",
                    "     /   \\",
                    r"    /_____\\",
                ]
            )
            title = f"{name} [defeated]"
        else:
            art = "\n".join(
                [
                    "   .-------.",
                    "  /  o   o  \\",
                    " |     ^     |",
                    r" |   \\___//  |",
                    r"  \\_________/",
                    r"     /| |\\",
                    r"    /_| |_\\",
                ]
            )
            title = f"{name} [{mood_text}]"
        return f"[b]{title}[/b]\n\n{art}"

    def make_ascii_art_placeholder(self, snapshot: dict[str, Any]) -> str:
        key = self.choose_ascii_art_key(snapshot)
        if key.startswith("npc::"):
            art = ASCII_ART_POLISHED.get(key)
            if art:
                name = snapshot.get("focus_npc_name") or "Unknown NPC"
                state = snapshot.get("focus_npc_state") or "alive"
                mood = snapshot.get("focus_npc_mood") or "unknown"
                title = f"{name} [{'defeated' if state == 'dead' else mood}]"
                return f"[b]{title}[/b]\n\n{art}"
            return self.make_npc_ascii_placeholder(
                snapshot.get("focus_npc_name") or "Unknown NPC",
                snapshot.get("focus_npc_state") or "alive",
                snapshot.get("focus_npc_mood"),
            )
        art = ASCII_ART_POLISHED.get(key, "")
        title = snapshot.get("location", "Scene") or "Scene"
        return f"[b]{title}[/b]\n\n{art}"

    def update_state(self, snapshot: dict[str, Any]) -> None:
        goal = self.query_one("#goal", Static)
        location = self.query_one("#location", Static)
        status = self.query_one("#status", Static)
        ascii_art = self.query_one("#ascii_art", Static)

        goal.update(f"[b]Goal[/b]\n{snapshot['goal']}")
        ascii_art.update(self.make_ascii_art_placeholder(snapshot))

        parts = [f"[b]{snapshot['location']}[/b]", "", snapshot["description"]]
        if snapshot["items"]:
            parts.extend(
                ["", "[b]Items here[/b]"] + [f"- {item}" for item in snapshot["items"]]
            )
        if snapshot["npcs"]:
            parts.extend(
                ["", "[b]People / creatures[/b]"]
                + [f"- {npc}" for npc in snapshot["npcs"]]
            )
        if snapshot["features"]:
            parts.extend(
                ["", "[b]Features[/b]"]
                + [f"- {feature}" for feature in snapshot["features"]]
            )
        location.update("\n".join(parts))

        attack_low, attack_high = snapshot["attack"]
        status_lines = [
            "[b]Player[/b]",
            f"HP: {snapshot['health']}/{snapshot['max_health']}",
            f"Attack: {attack_low}-{attack_high}",
            f"Defense: {snapshot['defense']}",
            f"Gold: {snapshot['gold']}",
            f"Bounty: {snapshot['bounty']}",
            "",
            "[b]Equipment[/b]",
        ]
        for slot, item_name in snapshot["equipment"].items():
            status_lines.append(f"- {slot}: {item_name}")
        status_lines.extend(["", "[b]Inventory[/b]"])
        if snapshot["inventory"]:
            for item_name, count in snapshot["inventory"]:
                suffix = f" x{count}" if count > 1 else ""
                status_lines.append(f"- {item_name}{suffix}")
        else:
            status_lines.append("- empty")
        status.update("\n".join(status_lines))

    def clear_options(self) -> None:
        option_list = self.query_one("#options", OptionList)
        option_list.clear_options()
        self.pending_options = []

    def show_choices(self, prompt: str, options: list[str]) -> None:
        self.interaction_mode = "choice"
        self.pending_options = list(options)
        self.query_one("#prompt", Static).update(
            f"{prompt}\n\nUse ↑/↓ and Enter, number keys 1-9, or click."
        )
        self.hide_text_input()
        option_list = self.query_one("#options", OptionList)
        option_list.clear_options()
        display_options = [f"{i}. {label}" for i, label in enumerate(options, start=1)]
        option_list.add_options(display_options)
        option_list.highlighted = 0 if options else None
        option_list.focus()

    def show_text_input(self, prompt: str) -> None:
        self.interaction_mode = "text"
        self.clear_options()
        self.query_one("#prompt", Static).update(prompt)
        input_widget = self.query_one("#text_input", Input)
        input_widget.remove_class("hidden")
        input_widget.value = ""
        input_widget.placeholder = prompt or "Type response and press Enter"
        input_widget.focus()

    def hide_text_input(self) -> None:
        input_widget = self.query_one("#text_input", Input)
        input_widget.add_class("hidden")
        input_widget.value = ""

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.interaction_mode != "choice":
            return
        self.archive_current_event()
        self.bridge.responses.put(event.option_index)
        self.clear_options()
        self.query_one("#prompt", Static).update("Resolving action...")

    def on_key(self, event: Key) -> None:
        if self.interaction_mode != "choice":
            return

        option_list = self.query_one("#options", OptionList)
        count = len(self.pending_options)
        if count == 0:
            return

        key = event.key
        option_list_has_focus = self.focused is option_list

        # When the OptionList itself has focus, let it handle arrow keys and
        # Enter so we do not double-advance the selection.
        if option_list_has_focus and key in {"up", "down", "enter"}:
            return

        if key == "up":
            current = option_list.highlighted
            if current is None:
                option_list.highlighted = 0
            else:
                option_list.highlighted = max(0, current - 1)
            event.stop()
            return

        if key == "down":
            current = option_list.highlighted
            if current is None:
                option_list.highlighted = 0
            else:
                option_list.highlighted = min(count - 1, current + 1)
            event.stop()
            return

        if key == "enter":
            current = option_list.highlighted
            if current is None:
                current = 0
            self.archive_current_event()
            self.bridge.responses.put(current)
            self.clear_options()
            self.query_one("#prompt", Static).update("Resolving action...")
            event.stop()
            return

        if len(key) == 1 and key.isdigit():
            index = int(key) - 1
            if 0 <= index < count:
                option_list.highlighted = index
                self.archive_current_event()
                self.bridge.responses.put(index)
                self.clear_options()
                self.query_one("#prompt", Static).update("Resolving action...")
                event.stop()
                return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "text_input" or self.interaction_mode != "text":
            return
        self.archive_current_event()
        self.bridge.responses.put(event.value)
        self.hide_text_input()
        self.query_one("#prompt", Static).update("Resolving input...")


if __name__ == "__main__":
    RPGTextualApp().run()
