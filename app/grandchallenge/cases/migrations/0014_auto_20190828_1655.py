# Generated by Django 2.2.4 on 2019-08-28 16:55

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("cases", "0013_rawimageuploadsession_reader_study")]

    operations = [
        migrations.AlterField(
            model_name="image",
            name="study",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="studies.Study",
            ),
        ),
        migrations.AlterField(
            model_name="imagefile",
            name="image",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="files",
                to="cases.Image",
            ),
        ),
    ]
