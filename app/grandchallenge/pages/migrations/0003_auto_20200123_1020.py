# Generated by Django 2.2.9 on 2020-01-23 10:20

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pages", "0002_auto_20181101_0933"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="page", options={"ordering": ["challenge", "order"]},
        ),
        migrations.RenameField(
            model_name="page",
            old_name="permission_lvl",
            new_name="permission_level",
        ),
        migrations.AlterField(
            model_name="page",
            name="challenge",
            field=models.ForeignKey(
                help_text="Which challenge does this page belong to?",
                on_delete=django.db.models.deletion.CASCADE,
                to="challenges.Challenge",
            ),
        ),
    ]