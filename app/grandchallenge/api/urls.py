from django.conf import settings
from django.conf.urls import include, url
from django.urls import path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions, routers

from grandchallenge.algorithms.views import (
    AlgorithmImageViewSet,
    AlgorithmViewSet,
    JobViewSet,
    ResultViewSet,
)
from grandchallenge.archives.views import ArchiveViewSet
from grandchallenge.cases.views import (
    ImageViewSet,
    RawImageFileViewSet,
    RawImageUploadSessionViewSet,
)
from grandchallenge.jqfileupload.views import StagedFileViewSet
from grandchallenge.reader_studies.views import (
    AnswerViewSet,
    QuestionViewSet,
    ReaderStudyViewSet,
)
from grandchallenge.retina_api.views import (
    ImageLevelAnnotationsForImageViewSet,
    LandmarkAnnotationSetViewSet,
    PathologyAnnotationViewSet,
    QualityAnnotationViewSet,
    RetinaPathologyAnnotationViewSet,
    TextAnnotationViewSet,
)
from grandchallenge.statistics.views import MetricsAPIView
from grandchallenge.subdomains.utils import reverse_lazy
from grandchallenge.workstation_configs.views import WorkstationConfigViewSet
from grandchallenge.workstations.views import SessionViewSet

app_name = "api"

router = routers.DefaultRouter()

# Algorithms
router.register(
    r"algorithms/images", AlgorithmImageViewSet, basename="algorithms-image"
)
router.register(r"algorithms/jobs", JobViewSet, basename="algorithms-job")
router.register(
    r"algorithms/results", ResultViewSet, basename="algorithms-result"
)
router.register(r"algorithms", AlgorithmViewSet, basename="algorithm")

# Archives
router.register(r"archives", ArchiveViewSet, basename="archive")

# Cases
router.register(r"cases/images", ImageViewSet, basename="image")
router.register(
    r"cases/upload-sessions/files",
    RawImageFileViewSet,
    basename="upload-session-file",
)
router.register(
    r"cases/upload-sessions",
    RawImageUploadSessionViewSet,
    basename="upload-session",
)

# Chunked uploads
router.register(r"chunked-uploads", StagedFileViewSet, basename="staged-file")

# Reader studies
router.register(
    r"reader-studies/answers", AnswerViewSet, basename="reader-studies-answer"
)
router.register(
    r"reader-studies/questions",
    QuestionViewSet,
    basename="reader-studies-question",
)
router.register(r"reader-studies", ReaderStudyViewSet, basename="reader-study")

# Retina
router.register(
    r"retina/landmark-annotation",
    LandmarkAnnotationSetViewSet,
    basename="retina-landmark-annotation",
)
router.register(
    r"retina/image-level-annotation-for-image",
    ImageLevelAnnotationsForImageViewSet,
    basename="retina-image-level-annotation-for-image",
)
router.register(
    r"retina/quality-annotation",
    QualityAnnotationViewSet,
    basename="retina-quality-annotation",
)
router.register(
    r"retina/pathology-annotation",
    PathologyAnnotationViewSet,
    basename="retina-pathology-annotation",
)
router.register(
    r"retina/retina-pathology-annotation",
    RetinaPathologyAnnotationViewSet,
    basename="retina-retina-pathology-annotation",
)
router.register(
    r"retina/text-annotation",
    TextAnnotationViewSet,
    basename="retina-text-annotation",
)

# Workstations
router.register(
    r"workstations/configs",
    WorkstationConfigViewSet,
    basename="workstations-config",
)
router.register(r"workstations/sessions", SessionViewSet)

schema_view = get_schema_view(
    openapi.Info(
        title=f"{settings.SESSION_COOKIE_DOMAIN.lstrip('.')} API",
        default_version="v1",
        description=f"The API for {settings.SESSION_COOKIE_DOMAIN.lstrip('.')}.",
        license=openapi.License(name="Apache License 2.0"),
        terms_of_service=reverse_lazy(
            "policies:detail", kwargs={"slug": "terms-of-service"}
        ),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
    patterns=[
        path("api/v1/", include(router.urls)),
        path("api/v1/metrics/", MetricsAPIView.as_view()),
    ],
)

urlpatterns = [
    url(
        r"^swagger(?P<format>\.json|\.yaml)$",
        schema_view.without_ui(),
        name="schema-json",
    ),
    # Do not namespace the router.urls without updating the view names in
    # the serializers
    path("v1/", include(router.urls)),
    path("v1/metrics/", MetricsAPIView.as_view(), name="metrics"),
    path("auth/", include("rest_framework.urls", namespace="rest_framework")),
    path("", schema_view.with_ui("swagger"), name="schema-docs"),
]
