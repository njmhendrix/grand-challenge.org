from django.urls import include, path
from rest_framework.routers import SimpleRouter

from grandchallenge.retina_api import views

app_name = "retina_api"

annotation_router = SimpleRouter()
annotation_router.register(
    "singlepolygonannotation",
    views.LegacySinglePolygonViewSet,
    basename="singlepolygonannotation",
)
annotation_router.register(
    "polygonannotationset",
    views.LegacyPolygonAnnotationSetViewSet,
    basename="polygonannotationset",
)
annotation_router.register(
    "etdrsgridannotation",
    views.ETDRSGridAnnotationViewSet,
    basename="etdrsgridannotation",
)
annotation_router.register(
    "imagequalityannotation",
    views.ImageQualityAnnotationViewSet,
    basename="imagequalityannotation",
)
annotation_router.register(
    "imagepathologyannotation",
    views.ImagePathologyAnnotationViewSet,
    basename="imagepathologyannotation",
)
annotation_router.register(
    "retinaimagepathologyannotation",
    views.RetinaImagePathologyAnnotationViewSet,
    basename="retinaimagepathologyannotation",
)
annotation_router.register(
    "imagetextannotation",
    views.ImageTextAnnotationViewSet,
    basename="imagetextannotation",
)
urlpatterns = [
    path("archives/", views.ArchiveView.as_view(), name="archives-api-view"),
    path(
        "archive_data/",
        views.ArchiveAPIView.as_view(),
        name="archive-data-api-view",
    ),
    path(
        "archive_data/<uuid:pk>/",
        views.ArchiveAPIView.as_view(),
        name="archive-data-api-view",
    ),
    path(
        "image/<str:image_type>/<str:patient_identifier>/<str:study_identifier>/<str:image_identifier>/<str:image_modality>/",
        views.ImageView.as_view(),
        name="image-api-view",
    ),
    path(
        "data/<str:data_type>/<int:user_id>/<str:archive_identifier>/<str:patient_identifier>/",
        views.DataView.as_view(),
        name="data-api-view",
    ),
    path(
        "annotation/polygon/<int:user_id>/<uuid:image_id>/",
        views.PolygonListView.as_view(),
        name="polygon-annotation-list-view",
    ),
    path(
        "annotation/polygon/users/<uuid:image_id>",
        views.GradersWithPolygonAnnotationsListView.as_view(),
        name="polygon-annotation-users-list-view",
    ),
    path(
        "annotation/landmark/users/<int:user_id>",
        views.LandmarkAnnotationSetForImageList.as_view(),
        name="landmark-annotation-images-list-view",
    ),
    path(
        "registration/octobs/<uuid:image_id>",
        views.OctObsRegistrationRetrieve.as_view(),
        name="octobs-registration-detail-view",
    ),
    path(
        "image/<uuid:image_id>/spacing/",
        views.ImageElementSpacingView.as_view(),
        name="image-element-spacing-view",
    ),
    path("annotation/<int:user_id>/", include(annotation_router.urls)),
    path(
        "image/thumbnail/<uuid:pk>/",
        views.B64ThumbnailAPIView.as_view(),
        name="image-thumbnail",
    ),
    path(
        "image/thumbnail/<uuid:pk>/<int:width>/<int:height>/",
        views.B64ThumbnailAPIView.as_view(),
        name="image-thumbnail",
    ),
]
