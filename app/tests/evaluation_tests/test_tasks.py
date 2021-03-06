from pathlib import Path

import docker
import pytest
from django.core.exceptions import ValidationError

from grandchallenge.components.tasks import validate_docker_image
from grandchallenge.evaluation.models import Method
from tests.factories import MethodFactory, SubmissionFactory


@pytest.mark.django_db
def test_submission_evaluation(
    client, evaluation_image, submission_file, settings
):
    # Override the celery settings
    settings.task_eager_propagates = (True,)
    settings.task_always_eager = (True,)

    # Upload a submission and create an evaluation
    dockerclient = docker.DockerClient(
        base_url=settings.COMPONENTS_DOCKER_BASE_URL
    )

    eval_container, sha256 = evaluation_image

    method = MethodFactory(
        image__from_path=eval_container, image_sha256=sha256, ready=True
    )

    # We should not be able to download methods
    with pytest.raises(NotImplementedError):
        _ = method.image.url

    num_containers_before = len(dockerclient.containers.list())
    num_volumes_before = len(dockerclient.volumes.list())

    # This will create an evaluation, and we'll wait for it to be executed
    submission = SubmissionFactory(
        file__from_path=submission_file, challenge=method.challenge
    )

    # The evaluation method should clean up after itself
    assert len(dockerclient.volumes.list()) == num_volumes_before
    assert len(dockerclient.containers.list()) == num_containers_before

    # The evaluation method should return the correct answer
    assert len(submission.evaluation_set.all()) == 1
    assert (
        submission.evaluation_set.first()
        .outputs.get(interface__slug="metrics-json-file")
        .value["acc"]
        == 0.5
    )

    # Try with a csv file
    submission = SubmissionFactory(
        file__from_path=Path(__file__).parent / "resources" / "submission.csv",
        challenge=method.challenge,
    )

    assert len(submission.evaluation_set.all()) == 1
    assert (
        submission.evaluation_set.first()
        .outputs.get(interface__slug="metrics-json-file")
        .value["acc"]
        == 0.5
    )


@pytest.mark.django_db
def test_method_validation(evaluation_image):
    """The validator should set the correct sha256 and set the ready bit."""
    container, sha256 = evaluation_image
    method = MethodFactory(image__from_path=container)

    # The method factory fakes the sha256 on creation
    assert method.image_sha256 != sha256
    assert method.ready is False

    validate_docker_image(
        pk=method.pk,
        app_label=method._meta.app_label,
        model_name=method._meta.model_name,
    )

    method = Method.objects.get(pk=method.pk)
    assert method.image_sha256 == sha256
    assert method.ready is True


@pytest.mark.django_db
def test_method_validation_invalid_dockerfile(alpine_images):
    """Uploading two images in a tar archive should fail."""
    method = MethodFactory(image__from_path=alpine_images)
    assert method.ready is False

    with pytest.raises(ValidationError):
        validate_docker_image(
            pk=method.pk,
            app_label=method._meta.app_label,
            model_name=method._meta.model_name,
        )

    method = Method.objects.get(pk=method.pk)
    assert method.ready is False
    assert "should only have 1 image" in method.status


@pytest.mark.django_db
def test_method_validation_root_dockerfile(root_image):
    """Uploading two images in a tar archive should fail."""
    method = MethodFactory(image__from_path=root_image)
    assert method.ready is False

    with pytest.raises(ValidationError):
        validate_docker_image(
            pk=method.pk,
            app_label=method._meta.app_label,
            model_name=method._meta.model_name,
        )

    method = Method.objects.get(pk=method.pk)
    assert method.ready is False
    assert "runs as root" in method.status


@pytest.mark.django_db
def test_method_validation_not_a_docker_tar(submission_file):
    """Upload something that isn't a docker file should be invalid."""
    method = MethodFactory(image__from_path=submission_file)
    assert method.ready is False

    with pytest.raises(ValidationError):
        validate_docker_image(
            pk=method.pk,
            app_label=method._meta.app_label,
            model_name=method._meta.model_name,
        )

    method = Method.objects.get(pk=method.pk)
    assert method.ready is False
    assert "manifest.json not found" in method.status
