from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def api_root(request):
    return JsonResponse({
        'status': 'ok',
        'message': 'API PCA est en ligne',
        'endpoints': {
            'upload_csv': '/api/upload-csv/',
            'perform_pca': '/api/perform-pca/',
        }
    })

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('backend.urls')),
    path('', api_root),
]
