from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="IPAddressField",
            fields=[
                (
                    "id",
                    models.AutoField(
                        verbose_name="ID",
                        serialize=False,
                        auto_created=True,
                        primary_key=True,
                    ),
                ),
                ("ip", models.IPAddressField(null=True, blank=True)),
            ],
        ),
    ]

from django.test import TestCase
from django.db import connection

class TestCollationPropagation(TestCase):
    def test_foreign_key_collation_propagation_repro(self):
        # Simulate the scenario where Account.id has db_collation set
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE `b_manage_account` (
                    `id` varchar(22) COLLATE `utf8_bin` NOT NULL,
                    PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            cursor.execute("""
                CREATE TABLE `b_manage_address` (
                    `id` varchar(22) NOT NULL,
                    `account_id` varchar(22) NOT NULL,
                    PRIMARY KEY (`id`),
                    FOREIGN KEY (`account_id`) REFERENCES `b_manage_account`(`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            cursor.execute("""
                CREATE TABLE `b_manage_profile` (
                    `id` varchar(22) NOT NULL,
                    `account_id` varchar(22) NULL,
                    PRIMARY KEY (`id`),
                    FOREIGN KEY (`account_id`) REFERENCES `b_manage_account`(`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)

        # Check if the foreign keys were created with the correct collation
        cursor.execute("SHOW CREATE TABLE b_manage_address")
        address_table_create = cursor.fetchone()[1]
        self.assertIn('COLLATE `utf8_bin`', address_table_create)

        cursor.execute("SHOW CREATE TABLE b_manage_profile")
        profile_table_create = cursor.fetchone()[1]
        self.assertIn('COLLATE `utf8_bin`', profile_table_create)
