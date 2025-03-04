from django.urls import path
from . import views

urlpatterns = [
    path('upload-csv-data/', views.upload_csv_data, name='upload_csv_data'),
    path('perform-pca/', views.perform_pca, name='perform_pca'),
    path('test-example/', views.test_example, name='test_example'),
] 