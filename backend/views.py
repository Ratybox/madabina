from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from .utils import pca, prepare_response

uploaded_csv_data = {}


@api_view(['POST'])
def upload_csv_data(request):
    try:
        if 'file' not in request.FILES:
            return Response({"error": "Aucun fichier n'a été envoyé."}, status=status.HTTP_400_BAD_REQUEST)

        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file, encoding='utf-8')
        uploaded_csv_data['df'] = df 
        
        return Response({
            "message": "Fichier CSV uploadé avec succès",
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def perform_pca(request):

    try:
        if 'df' not in uploaded_csv_data:
            return Response({"error": "Aucune donnée CSV n'a été chargée. Veuillez d'abord charger un fichier CSV."}, 
                           status=status.HTTP_400_BAD_REQUEST)
        
        df = uploaded_csv_data['df']
        
        pca_type = request.data.get('pca_type', 'normalized')
        
        if pca_type not in ['normalized', 'homogeneous', 'heterogeneous']:
            return Response({"error": "Type de PCA non valide. Utilisez 'normalized', 'homogeneous' ou 'heterogeneous'."}, 
                           status=status.HTTP_400_BAD_REQUEST)
        
        results = pca(df, pca_type=pca_type)
        
        return Response(prepare_response(results), status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['GET'])

def test_example(request):

    try:

        data = np.array([[16, 20, 12], [20, 12, 22], [16, 24, 26], [28, 24, 20]])

        results = {

            'normalized': pca(data, pca_type='normalized'),

            'homogeneous': pca(data, pca_type='homogeneous'),

            'heterogeneous': pca(data, pca_type='heterogeneous')

        }

        return Response({

            'message': 'Test effectué avec succès',

            'data': data.tolist(),

            'results': {k: prepare_response(v) for k, v in results.items()}

        }, status=status.HTTP_200_OK)

    except Exception as e:

        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)