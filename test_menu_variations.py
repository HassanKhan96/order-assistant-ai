"""Exercise configurable items across normalization and validation."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
import unittest

from menu_normalization import normalize_api_menu
from menu_validation import validate_menu


def choice(identifier, name, price=0):
    return dict(id=identifier, name=name, price=price, availability="AVAILABLE",
                is_preselected=False, position="ignored")


def group(identifier, name, choices=(), selections=(), optional=False):
    return dict(id=identifier, name=name, type="PRODUCT_BASED" if selections else "CUSTOM",
                choices=list(choices), sub_product_selection=list(selections),
                is_optional=optional, min_selection=1, max_selection=1,
                availability="AVAILABLE")


def item(identifier, name, category="mains", groups=()):
    return dict(id=identifier, name=name, category_id=category,
                availability="AVAILABLE", sizes=[], menu_item_variations=list(groups),
                pricing_and_availabilities=[dict(channel="ONLINE_DELIVERY", price=10,
                                                availability="AVAILABLE")])


def normalize(items):
    payload = SimpleNamespace(
        menu=items,
        categories=[dict(id="mains", name="Mains"), dict(id="sides", name="Sides")],
        menu_endpoint="items-url", categories_endpoint="categories-url",
        fetched_at=datetime.now(timezone.utc),
    )
    return normalize_api_menu(payload, menu_id="store", restaurant_name="Restaurant",
                              currency="GBP", channel="ONLINE_DELIVERY")


class MenuVariationTests(unittest.TestCase):
    def test_multiple_dimensions_and_size_preserve_price_components(self):
        record = item("pizza", "Pizza", groups=[
            group("meat", "Meat", [choice("chicken", "Chicken", 2), choice("lamb", "Lamb", 3)]),
            group("crust", "Crust", [choice("thin", "Thin", 0), choice("thick", "Thick", 1)]),
            group("sauce", "Sauce", [choice("green", "Green"), choice("red", "Red")]),
        ])
        record["sizes"] = [dict(id="large", name="Large", price=4, availability="AVAILABLE")]
        menu = normalize([record])
        result = menu.items[0]
        self.assertEqual(result.variants[0].price, Decimal("14"))
        self.assertEqual(result.variation_groups[0].choices[0].price_adjustment, Decimal("2"))
        self.assertEqual(len(result.variation_groups), 3)
        for text in ("Meat: Chicken, Lamb", "Crust: Thin, Thick", "Sauce: Green, Red"):
            self.assertIn(text, result.embedding_text)
        self.assertTrue(validate_menu(menu).is_valid)

    def test_product_groups_are_ignored_even_when_required_or_malformed(self):
        selection = dict(id="side-selection", category_id="sides",
                         hide_following_items=["chips"], void_selected_item_price=True)
        menu = normalize([
            item("meal", "Meal", groups=[group("side", "Side", selections=[selection])]),
            item("salad", "Salad", category="sides"),
            item("chips", "Chips", category="sides"),
        ])
        self.assertEqual(menu.items[0].variation_groups, ())
        self.assertNotIn("Side:", menu.items[0].embedding_text)
        self.assertTrue(validate_menu(menu).is_valid)
        malformed = dict(type="PRODUCT_BASED", sub_product_selection="ignored")
        menu = normalize([item("meal", "Meal", groups=[malformed])])
        self.assertTrue(validate_menu(menu).is_valid)

    def test_required_empty_group_blocks_validation_but_optional_does_not(self):
        required = normalize([item("meal", "Meal", groups=[group("meat", "Meat")])])
        self.assertIn("unsatisfiable_selection", {e.code for e in validate_menu(required).errors})
        optional = normalize([item("meal", "Meal", groups=[group("sauce", "Sauce", optional=True)])])
        self.assertTrue(validate_menu(optional).is_valid)

    def test_shared_groups_are_valid_and_duplicates_within_group_are_rejected(self):
        shared = group("meat", "Meat", [choice("chicken", "Chicken")])
        menu = normalize([item("one", "One", groups=[shared]), item("two", "Two", groups=[shared])])
        self.assertTrue(validate_menu(menu).is_valid)
        original = menu.items[0].variation_groups[0]
        broken = replace(original, choices=original.choices * 2, min_selection=2, max_selection=1)
        bad_item = replace(menu.items[0], variation_groups=(broken,))
        codes = {e.code for e in validate_menu(replace(menu, items=(bad_item,))).errors}
        self.assertIn("duplicate_id", codes)
        self.assertIn("invalid_selection_limits", codes)

    def test_inactive_product_fields_do_not_pollute_custom_choice_embeddings(self):
        custom = group("crust", "Crust", [choice("thin", "Thin")])
        custom["sub_product_selection"] = "ignored even when malformed"
        menu = normalize([item("pizza", "Pizza", groups=[custom]), item("salad", "Salad", category="sides")])
        self.assertNotIn("Salad", menu.items[0].embedding_text)
        self.assertTrue(validate_menu(menu).is_valid)


if __name__ == "__main__":
    unittest.main()
