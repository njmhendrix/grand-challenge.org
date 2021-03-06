# Generated by Django 2.0.8 on 2018-08-08 13:38

import django.contrib.postgres.fields.citext
import django.db.models.deletion
from django.contrib.postgres.operations import CITextExtension
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("challenges", "0007_auto_20180802_1437")]

    operations = [
        CITextExtension(),
        migrations.CreateModel(
            name="BodyRegion",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "region",
                    django.contrib.postgres.fields.citext.CICharField(
                        max_length=16, unique=True
                    ),
                ),
            ],
            options={"ordering": ("region",)},
        ),
        migrations.CreateModel(
            name="BodyStructure",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "structure",
                    django.contrib.postgres.fields.citext.CICharField(
                        max_length=16, unique=True
                    ),
                ),
                (
                    "region",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="challenges.BodyRegion",
                    ),
                ),
            ],
            options={"ordering": ("region", "structure")},
        ),
        migrations.CreateModel(
            name="ImagingModality",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "modality",
                    django.contrib.postgres.fields.citext.CICharField(
                        max_length=16, unique=True
                    ),
                ),
            ],
            options={"ordering": ("modality",)},
        ),
        migrations.CreateModel(
            name="TaskType",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "type",
                    django.contrib.postgres.fields.citext.CICharField(
                        max_length=16, unique=True
                    ),
                ),
            ],
            options={"ordering": ("type",)},
        ),
        migrations.RenameField(
            model_name="challenge", old_name="created_at", new_name="created"
        ),
        migrations.RenameField(
            model_name="externalchallenge",
            old_name="created_at",
            new_name="created",
        ),
        migrations.AddField(
            model_name="challenge",
            name="data_license_agreement",
            field=models.TextField(
                blank=True,
                help_text="What is the data license agreement for this challenge?",
            ),
        ),
        migrations.AddField(
            model_name="challenge",
            name="modified",
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name="challenge",
            name="number_of_test_cases",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="challenge",
            name="number_of_training_cases",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="data_license_agreement",
            field=models.TextField(
                blank=True,
                help_text="What is the data license agreement for this challenge?",
            ),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="data_stored",
            field=models.BooleanField(
                default=False,
                help_text="Has the grand-challenge team stored the data?",
            ),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="modified",
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="number_of_test_cases",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="number_of_training_cases",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="challenge",
            name="modalities",
            field=models.ManyToManyField(
                blank=True,
                help_text="What imaging modalities are used in this challenge?",
                to="challenges.ImagingModality",
            ),
        ),
        migrations.AddField(
            model_name="challenge",
            name="structures",
            field=models.ManyToManyField(
                blank=True,
                help_text="What structures are used in this challenge?",
                to="challenges.BodyStructure",
            ),
        ),
        migrations.AddField(
            model_name="challenge",
            name="task_types",
            field=models.ManyToManyField(
                blank=True,
                help_text="What type of task is this challenge?",
                to="challenges.TaskType",
            ),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="modalities",
            field=models.ManyToManyField(
                blank=True,
                help_text="What imaging modalities are used in this challenge?",
                to="challenges.ImagingModality",
            ),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="structures",
            field=models.ManyToManyField(
                blank=True,
                help_text="What structures are used in this challenge?",
                to="challenges.BodyStructure",
            ),
        ),
        migrations.AddField(
            model_name="externalchallenge",
            name="task_types",
            field=models.ManyToManyField(
                blank=True,
                help_text="What type of task is this challenge?",
                to="challenges.TaskType",
            ),
        ),
    ]
