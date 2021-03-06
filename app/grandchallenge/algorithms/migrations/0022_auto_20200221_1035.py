# Generated by Django 3.0.2 on 2020-02-21 10:35

from django.db import migrations


def consolidate_info_reverse(*_, **__):
    pass


def consolidate_info_forward(apps, schema_editor):
    Algorithm = apps.get_model("algorithms", "Algorithm")  # noqa: N806

    for alg in Algorithm.objects.all():
        detail_markdown = alg.description

        if alg.contact_information:
            detail_markdown += (
                f"\n\n### Contact Information\n\n{alg.contact_information}"
            )

        if alg.additional_information:
            detail_markdown += f"\n\n### Additional Information\n\n{alg.additional_information}"

        alg.detail_page_markdown = detail_markdown
        alg.save()


class Migration(migrations.Migration):

    dependencies = [
        ("algorithms", "0021_auto_20200221_1034"),
    ]

    operations = [
        migrations.RunPython(
            consolidate_info_forward, consolidate_info_reverse,
        )
    ]
