import os
import dotenv
from supabase import create_client
from src.utils import Chunking, indexation_chroma, chroma_config, main_rag

""" Chargement des variables d'nevironnement depuis le fichier .env """
dotenv.load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
Supabase_Client = create_client(url, key)

gemini_key = os.getenv("GEMINI_API_KEY")

def main():

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

    query = input("Entrez votre question : ")

    main_rag(query[0])

if __name__ == "__main__":
    main()