import docker
from django.conf import settings

from grandchallenge.components.backends.docker import Service


def test_service_start_cleanup():
    job_id = "12345"
    job_model = "test"
    exec_image = "crccheck/hello-world"
    filters = {"label": f"job={job_model}-{job_id}"}

    dockerclient = docker.DockerClient(
        base_url=settings.COMPONENTS_DOCKER_BASE_URL
    )
    dockerclient.images.pull(exec_image)

    exec_sha256 = dockerclient.images.get(exec_image).id

    s = Service(
        job_id=job_id,
        job_model=job_model,
        exec_image=None,
        exec_image_sha256=exec_sha256,
    )
    assert len(dockerclient.containers.list(filters=filters)) == 0

    try:
        s.start(http_port=80, websocket_port=81, hostname="test-local")

        containers = dockerclient.containers.list(filters=filters)
        assert len(containers) == 1

        labels = containers[0].labels

        expected_labels = {
            "job": f"{job_model}-{job_id}",
            "traefik.enable": "true",
            "traefik.http.routers.test-local-http.entrypoints": "workstation-http",
            "traefik.http.routers.test-local-http.rule": "Host(`test-local`)",
            "traefik.http.routers.test-local-http.service": "test-local-http",
            "traefik.http.routers.test-local-websocket.entrypoints": "workstation-websocket",
            "traefik.http.routers.test-local-websocket.rule": "Host(`test-local`)",
            "traefik.http.routers.test-local-websocket.service": "test-local-websocket",
            "traefik.http.services.test-local-http.loadbalancer.server.port": "80",
            "traefik.http.services.test-local-websocket.loadbalancer.server.port": "81",
        }

        for k, v in expected_labels.items():
            assert labels[k] == v
    finally:
        s.stop_and_cleanup()
        assert len(dockerclient.containers.list(filters=filters)) == 0
