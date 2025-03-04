from django.urls import path
from . import views

urlpatterns = [
    path('upload-csv-data/', views.upload_csv_data, name='upload_csv_data'),
    path('get-data/', views.get_data, name='get_data'),
    path('perform-pca/', views.perform_pca, name='perform_pca'),
    path('perform-pca-homogeneous/', views.perform_pca_homogeneous, name='perform_pca_homogeneous'),
    path('perform-pca-heterogeneous/', views.perform_pca_heterogeneous, name='perform_pca_heterogeneous'),
] 