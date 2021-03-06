# Generated by Django 2.2.9 on 2020-01-20 17:53

from django.db import migrations, models

import grandchallenge.core.storage
import grandchallenge.uploads.models


class Migration(migrations.Migration):

    dependencies = [
        ("uploads", "0002_summernoteattachment"),
    ]

    operations = [
        migrations.AlterField(
            model_name="summernoteattachment",
            name="file",
            field=models.FileField(
                storage=grandchallenge.core.storage.PublicS3Storage(),
                upload_to=grandchallenge.uploads.models.summernote_upload_filepath,
            ),
        ),
    ]
