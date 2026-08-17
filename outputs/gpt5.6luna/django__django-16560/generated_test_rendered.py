from django.core.exceptions import ValidationError
from django.test import TestCase, skipUnlessDBFeature

from .models import (
    ChildProduct,
    ChildUniqueConstraintProduct,
    Product,
    UniqueConstraintConditionProduct,
    UniqueConstraintProduct,
)


class PerformConstraintChecksTest(TestCase):
    @skipUnlessDBFeature("supports_table_check_constraints")
    def test_full_clean_with_check_constraints_gate(self):
        product = Product(price=10, discounted_price=15)
        with self.assertRaises(ValidationError) as cm:
            product.full_clean()
        self.assertEqual(
            cm.exception.message_dict,
            {
                "__all__": [
                    "Constraint “price_gt_discounted_price_validation” is violated."
                ]
            },
        )

    @skipUnlessDBFeature("supports_table_check_constraints")
    def test_full_clean_with_check_constraints_gate_on_child_model_gate(self):
        product = ChildProduct(price=10, discounted_price=15)
        with self.assertRaises(ValidationError) as cm:
            product.full_clean()
        self.assertEqual(
            cm.exception.message_dict,
            {
                "__all__": [
                    "Constraint “price_gt_discounted_price_validation” is violated."
                ]
            },
        )

    @skipUnlessDBFeature("supports_table_check_constraints")
    def test_full_clean_with_check_constraints_gate_disabled_gate(self):
        product = Product(price=10, discounted_price=15)
        product.full_clean(validate_constraints=False)

    def test_full_clean_with_unique_constraints_gate(self):
        UniqueConstraintProduct.objects.create(name="product", color="yellow", rank=1)
        tests = [
            UniqueConstraintProduct(name="product", color="yellow", rank=1),
            # Child model.
            ChildUniqueConstraintProduct(name="product", color="yellow", rank=1),
        ]
        for product in tests:
            with self.subTest(model=product.__class__.__name__):
                with self.assertRaises(ValidationError) as cm:
                    product.full_clean()
                self.assertEqual(
                    cm.exception.message_dict,
                    {
                        "__all__": [
                            "Unique constraint product with this Name and Color "
                            "already exists."
                        ],
                        "rank": [
                            "Unique constraint product with this Rank already exists."
                        ],
                    },
                )

    def test_full_clean_with_unique_constraints_gate_disabled_gate(self):
        UniqueConstraintProduct.objects.create(name="product", color="yellow", rank=1)
        product = UniqueConstraintProduct(name="product", color="yellow", rank=1)
        product.full_clean(validate_constraints=False)

    @skipUnlessDBFeature("supports_partial_indexes")
    def test_full_clean_with_partial_unique_constraints_gate(self):
        UniqueConstraintConditionProduct.objects.create(name="product")
        product = UniqueConstraintConditionProduct(name="product")
        with self.assertRaises(ValidationError) as cm:
            product.full_clean()
        self.assertEqual(
            cm.exception.message_dict,
            {
                "__all__": [
                    "Constraint “name_without_color_uniq_validation” is violated."
                ]
            },
        )

    @skipUnlessDBFeature("supports_partial_indexes")
    def test_full_clean_with_partial_unique_constraints_gate_disabled_gate(self):
        UniqueConstraintConditionProduct.objects.create(name="product")
        product = UniqueConstraintConditionProduct(name="product")
        product.full_clean(validate_constraints=False)

from django.test import TestCase, SimpleTestCase
from django.db.models.constraints import BaseConstraint

class TestBaseConstraintValidation(TestCase):
    def test_validate_without_custom_code_repro(self):
        constraint = BaseConstraint(name='test_constraint')
        try:
            constraint.validate(None)
        except ValidationError as e:
            self.assertIsNone(e.code)
