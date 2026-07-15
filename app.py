"""
    FAST API:

    Une API permet à un app de consommer des données sans avoir à gérer les accès à la base de données.
    FAST API permet de créer des api restful.

    Une api rest fonctionne avec des requête HTTP, c'est un protocole qui permet de communiquer avec le serveur.
    Il y a différents types de requêtes HTTP, qui dépendent de l'action que l'on veut effectuer sur le serveur. 
    - POST : créer un nouvelle element dans la base de données
    - GET : sert à récupérer des données depuis le serveur (celles par défaut)
    - PUT : modifier un éléméne texistant dans la base de données
    - DELETE : supprimer des items dans la base de données

    la serveur ne fait que détecter le type de requête, pour que l'action attentue soit effectuée, il faut associer la requ^te à un code python qui décrit l'action.
    le serveur détecte la type puis déclenceh le code python associé à la requête.
    on est donc responsable du respect des conventions PUT GET POST DELETE.

    pour pouvoir faire tourner Fast API il faut installe : uvicorn, qui est un serveur ASGI (Asynchronous Server Gateway Interface) qui permet de faire tourner des applications web asynchrones.

    
    200 OK : la ressource a été récupérée avec succès.
201 Created : une nouvelle ressource a été créée.
204 No Content : succès sans retour de contenu.
404 Not Found : la ressource demandée n'existe pas.
500 Internal Server Error : une erreur s’est produite côté serveur
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import dotenv
from supabase import create_client
from src.utils import Chunking, indexation_chroma, chroma_config, main_rag

dotenv.load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
Supabase_Client = create_client(url, key)

gemini_key = os.getenv("GEMINI_API_KEY")

# on définit une logique qui doit être exécuter avant le démarage de l'application
# le code ne sera exécuté qu'une seule fois, avant que l'app ne reçoive des requêtes
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tout ce qui est AVANT le yield s'exécute au DÉMARRAGE (Startup)
    print("Démarrage du serveur...") 
    print("Initialisation des ressources nécessaires à l'application...")
    try :
        response = Supabase_Client.table("api_docs").select("id", "texte", "url", "titre").execute()
        if not response or len(response.data) != 77:
            raise Exception("Erreur lors de la récupération des données depuis Supabase ou e nombre de documents récupérés est incorrect.")
    except Exception as e:
        raise(f"Erreur de récupération des données : {e}")
    
    try :
        collection = chroma_config()
        try :
            # liste_ids, liste_documents, liste_metadatas = make_chuns(documents_bruts=response.data)
            liste_ids, liste_documents, liste_metadatas = Chunking(response.data).make_chuns()
            try :
                indexation_chroma(liste_ids, liste_documents, liste_metadatas,collection)
            except Exception as e:
                print(f"Erreur lors de l'indexation Chroma : {e}")
        except Exception as e:
            print(f"Erreur lors de make_chunks: {e}")
    except Exception as e:
        print(f"Erreur lors de la configuration Chroma : {e}")
    
    yield  # L'API s'arrête ici et fonctionne pour les utilisateurs
    
    # Tout ce qui est APRÈS le yield s'exécute à la FERMETURE (Shutdown)
    print("Fermeture du serveur...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"statu": "Opérationnel"}

class QueryRequest(BaseModel):
    query: str

@app.post("/query/", status_code=201)
def get_query(payload: QueryRequest):
    query = payload.query
    try:
        resultat_rag = main_rag(query)
        return {
            "status": "success",
            "message": f"Received query: {query}",
            "responses": resultat_rag["response"],
            "sources": resultat_rag["sources"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'exécution du RAG : {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)