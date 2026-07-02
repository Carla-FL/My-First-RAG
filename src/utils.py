import re
import chromadb
import trafilatura
import unicodedata
from google import genai
from google.genai import types
from chromadb.utils import embedding_functions


def scraper_page(url: str, id ) -> dict | None:
    """
    Télécharge une page et extrait son contenu textuel.
    Retourne None si l'extraction échoue.

    Args:
        url (str): URL de la page à scraper
        id (int): Identifiant unique pour la page

    Returns:
        dict | None: Dictionnaire contenant l'ID, l'URL, le titre et le texte extrait, ou None si l'extraction échoue.
    """
    # Télécharger le HTML
    html = trafilatura.fetch_url(url)
    if not html:
        print(f"Échec téléchargement: {url}")
        return None

    # Extraire le contenu (sans commentaires, avec tableaux)
    texte = trafilatura.extract(
        html,
        include_comments=True,
        include_tables=True,
        output_format="txt"
    )

    if not texte:
        print(f"Pas de contenu extractible: {url}")
        return None

    # Extraire un titre depuis l'URL
    titre = url[41:].replace("?hl=fr", "").replace("/", "_")

    return {
        "id": id,
        "url": url,
        "titre": titre,
        "texte": texte
    }

def nettoyage(texte: str) -> str:
    """
    Nettoie le texte en normalisant les caractères Unicode, en supprimant les espaces multiples, les lignes vides excessives et les barres obliques inverses.

    Args:
        texte (str): Le texte à nettoyer.

    Returns:
        str: Le texte nettoyé.
    """
    texte = unicodedata.normalize("NFC", texte)
    texte = re.sub(r'[ \t]+', ' ', texte)  # Espaces multiples
    texte = re.sub(r'\n{3,}', '\n\n', texte)  # Lignes vides excessives
    texte = re.sub(r'\\', '', texte)  # supprime les \
    texte = texte.strip()
    return texte

class Chunking():
    """
    Classe pour découper un texte en segments (chunks) de taille fixe avec chevauchement.

    Attributes:
        texte (str): Le texte à découper.
        chunk_size (int): Taille maximale de chaque chunk (en nombre de mots).
        overlap (int): Nombre de mots qui se chevauchent entre les chunks.

    Methods:
        chunker_fixe(chunk_size:int, overlap:int, texte:str) -> list:
            Découpe un texte en segments/chunks de taille fixe avec chevauchement.
        
        chunker_phrase(): à venir

        chunker_secion(): à venir

        chunker_semantique(): à venir
        
    """

    def __init__(self, documents:list):
        self.documents = documents
        pass

    # methode de chunking fixe
    def chunker_fixe(self, chunk_size:int, overlap:int, texte:str) -> list:
        """ 
        Découpe un texte en segments/ chunks de taille fixe avec chevauchement.

        Args:
            chunk_size (int): Taille maximale de chaque chunk (en nombre de mots).
            overlap (int): Nombre de mots qui se chevauchent entre les chunks.
            texte (str): Le texte à découper.

        Returns:
            list: Liste de chunks (segments de texte).
        """
        assert chunk_size > overlap, "La taille du chunk doit être supérieure au chevauchement."

        mots = texte.split()

        chunks = []
        start = 0
        step = chunk_size - overlap

        for i in range(start, len(mots), step):
            chunk = " ".join(mots[i:i + chunk_size])
            chunks.append(chunk)
            #print(chunk)

            if start + step >= len(mots):
                break
            start += step
        
        return chunks
    
    def chunker_phrase():
        pass

    def chunker_secion():
        pass

    def chunker_semantique():
        pass


    # implementation du chunking
    def make_chuns(self)-> tuple[list, list, list]:
        """

        Args:
            documents_bruts (list): Liste de dictionnaires contenant les documents bruts avec leurs métadonnées, issus du scraping.

        Returns:
            tuple[list, list, list]: Trois listes : 
                - liste_ids : Liste des IDs uniques pour chaque chunk.
                - liste_documents : Liste des textes des chunks.
                - liste_metadatas : Liste des métadonnées associées à chaque chunk.
        """

        liste_ids = []
        liste_documents = []
        liste_metadatas = []

        print("Application du chunking et préparation de l'indexation...")
        chunk_size = 200
        overlap = 50
        for doc in self.documents:
            # On applique ta fonction de découpage sur le texte brut du document
            cleaned_text = nettoyage(doc["texte"])
            chunks_du_document = self.chunker_fixe(chunk_size,overlap, cleaned_text)
            
            for index, texte_du_chunk in enumerate(chunks_du_document):
                # print(f"Document ID: {doc['id']}, Chunk {index}: {texte_du_chunk[:60]}...")
                # Création d'un ID unique par chunk (ex: doc_1_chunk_000, doc_1_chunk_001...)
                id_unique_chunk = f"doc_{doc['id']}_chunk_{index:03d}"
                
                liste_ids.append(id_unique_chunk)
                liste_documents.append(texte_du_chunk) # Le texte que Chroma va vectoriser
                
                # On sauvegarde les métadonnées pour que le LLM sache d'où vient l'info
                liste_metadatas.append({
                    "document_parent_id": doc["id"],
                    "url": doc["url"],
                    "titre": doc["titre"],
                    "chunk_index": index
                })
        return liste_ids, liste_documents, liste_metadatas
    

# configuration de chroma avec un modèle d'embedding multilingue et création d'une collection
def chroma_config():
    """
    Configure Chroma avec un modèle d'embedding multilingue et crée une collection pour stocker les documents, embeddings et métadonnées.
    """

    # client local non persistant
    chroma_client = chromadb.Client()

    # config embedding mutilingue
    fonction_multilingue = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # creat a collection for our documents, embeddings and metadatas 
    collection = chroma_client.get_or_create_collection(name="youtube_api_docs",
                                        embedding_function=fonction_multilingue)
    
    return collection

# injection des chunks dans la collection chroma
def indexation_chroma(liste_ids, liste_documents, liste_metadatas, collection):
    """
    Injecte les chunks vectorisés dans la collection Chroma.

    Args:
        liste_ids (list): liste des IDs uniques pour chaque chunk.
        liste_documents (list): liste des textes des chunks.
        liste_metadatas (list): liste des métadonnées associées à chaque chunk.
    """

    if liste_ids:
        print(f"Vectorisation multilingue et injection de {len(liste_ids)} chunks dans Chroma...")
        try:
            # add dociuments to the collection  
            collection.add(
                ids= liste_ids,
                documents= liste_documents,
                metadatas= liste_metadatas
            )
            print("Les documents Supabase sont vectorisés dans Chroma.")
        except Exception as e:
            print(f"Erreur lors de l'injection Chroma : {e}")
    else:
        print("Aucun chunk généré")
    

# recherche de documents pour contexte dans la collection chroma
def get_contexte (query:list, n:int=3) -> str:
    """
    La fonction get_contexte interroge la collection Chroma pour récupérer les documents les plus pertinents en fonction de la requête de l'utilisateur. 
    Elle retourne le contexte sous forme de texte concaténé et les métadonnées associées.

    Args:
        query (list): Liste de chaînes de caractères représentant la requête de l'utilisateur.
        n (int): Nombre de résultats pertinents à récupérer (par défaut 3).

    Returns:
        tuple: Un tuple contenant le contexte sous forme de texte concaténé et les métadonnées associées.
    """

    # vérifie que querry est une liste de chaînes de caractères
    assert isinstance(query, list), "Query doit être une liste."
    assert all(isinstance(item, str) for item in query), "Tous les éléments de la liste query doivent être des chaînes de caractères."
    # vérifie que query n'est pas vide
    assert len(query) > 0, "La liste de requête ne peut pas être vide."
    # vérifie que n est un entier positif superieur à 1
    assert isinstance(n, int) and n > 1, "n doit être un entier positif supérieur à 0."

    # on récupère la collection Chroma
    client = chromadb.Client()
    collection = client.get_collection(name="youtube_api_docs")

    # recherche des docuement pertinents pour la requête
    resultats = collection.query(
        query_texts=query,
        n_results=n
    )

    chunks_ = resultats["documents"][0]
    metadatas_= resultats["metadatas"][0]

    contexte = "\n\n".join(
        [f"[Extrait {i+1} (Source: {meta['url']})]\n{texte}" 
        for i, (texte, meta) in enumerate(zip(chunks_, metadatas_))]
    )

    return contexte, metadatas_

# augmentation du prompt avec le contexte et la question du développeur
def make_augmented_promt(contexte:str, query:str) -> str:
    """
    Renvoie un prompt augmenté en combinant le contexte de la documentation et la question du développeur.

    Args:
        contexte (str): l'ensemble des extraits de documentation pertinents pour la question du développeur.
        query (str): la question posée par le développeur.
    
    Returns:
        Le prompt final contenant le contexte et la question du développeur.
    """

    # Le prompt final fusionne le contexte et la question
    prompt_augmented = f"""CONTEXTE DE LA DOCUMENTATION OFFICIELLE :
    {contexte}

    QUESTION DU DÉVELOPPEUR :
    {query}
    """

    return prompt_augmented

# interrogation du modèle Gemini pour générer une réponse à partir du prompt augmenté
def rag_executer(prompt:str, temperature:float=0.5) -> str:
    """
    Interroge le modèle Gemini pour générer une réponse à partir du prompt augmenté.
    
    Args:
        prompt (str): Le prompt augmenté contenant le contexte et la question du développeur.
    
    Returns:
        reponse (str) : La réponse générée par le modèle Gemini.
    
    """

    system_instruction = """Tu es un ingénieur support technique expert. Ton but est d'aider les développeurs en répondant à leurs questions.
    Consignes strictes :
    1. Réponds en utilisant UNIQUEMENT les informations du contexte fourni.
    2. Si la réponse n'est pas dans le contexte, dis clairement et poliment : "Je suis désolé, mais la documentation fournie ne contient pas l'information pour répondre à cette question."
    3. Ne pas inventer, extrapoler ou utiliser tes connaissances générales. Réponds TOUJOURS en français.
    """

    client_gemini = genai.Client()

    reponse = client_gemini.models.generate_content(
        model = "gemini-3.5-flash",
        contents =prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=temperature)

    )

    return reponse

# implémentation de la fonction principale pour exécuter le RAG
def main_rag(query:str, n:int=3, temperature:float=0.5):
    """
    Implémente le processus de RAG (Retrieval-Augmented Generation) pour répondre à une question posée par un développeur.

    Args:
        query (str): La question posée par le développeur.
        n (int): Le nombre de documents pertinents à récupérer pour le contexte (par défaut 3).
    """
    contexte, metadatas = get_contexte(query=[query], n=n)

    prompt_augmented = make_augmented_promt(contexte, query)

    response = rag_executer(prompt_augmented, temperature=temperature)

    print("\n" + "="*40)
    print(f"Votre question:\n{query}")
    print("\n" + "="*40)
    print("RÉPONSE DU RAG :")
    print("="*40)
    print(response.text)
    print("="*40)

    print("\n🔗 SOURCES UTILISÉES :")
    urls_uniques = set([meta['url'] for meta in metadatas])
    for url in urls_uniques:
        print(f"- {url}")

    return {
        "response": response.text,
        "sources": urls_uniques
    }
    
