# Generated by Django 3.0.6 on 2020-06-22 13:14

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("components", "0001_initial"),
        ("cases", "0024_auto_20200525_0634"),
        ("algorithms", "0025_algorithmimage_queue_override"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="job", options={"ordering": ("created",)},
        ),
        migrations.AlterModelOptions(
            name="result", options={"ordering": ("created",)},
        ),
        migrations.AddField(
            model_name="algorithm",
            name="inputs",
            field=models.ManyToManyField(
                related_name="algorithm_inputs",
                to="components.ComponentInterface",
            ),
        ),
        migrations.AddField(
            model_name="algorithm",
            name="outputs",
            field=models.ManyToManyField(
                related_name="algorithm_outputs",
                to="components.ComponentInterface",
            ),
        ),
        migrations.AddField(
            model_name="job",
            name="comment",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="job",
            name="inputs",
            field=models.ManyToManyField(
                related_name="algorithms_jobs_as_input",
                to="components.ComponentInterfaceValue",
            ),
        ),
        migrations.AddField(
            model_name="job",
            name="outputs",
            field=models.ManyToManyField(
                related_name="algorithms_jobs_as_output",
                to="components.ComponentInterfaceValue",
            ),
        ),
        migrations.AddField(
            model_name="job",
            name="public",
            field=models.BooleanField(
                default=False,
                help_text="If True, allow anyone to view this result along with the input image. Otherwise, only the job creator and algorithm editor will have permission to view this result.",
            ),
        ),
        migrations.AlterField(
            model_name="job",
            name="image",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="cases.Image",
            ),
        ),
    ]
