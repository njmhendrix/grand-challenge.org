# Generated by Django 3.0.2 on 2020-02-27 17:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("evaluation", "0027_auto_20200121_1634"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="completed_at",
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name="job",
            name="started_at",
            field=models.DateTimeField(null=True),
        ),
    ]
