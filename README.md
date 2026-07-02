# Pipeline RAG Bilingue : Ingestion de Documentation API & Recherche Sémantique

Ce projet implémente un **RAG (Retrieval-Augmented Generation)** pour la docuementation de l'**API YouTube Data V3**. Il s'agit de la première étape (V0) vers un outil de support technique plus performant et destiné à être déployé à grande échelle.

Pour l'instant, ce projet fonctionne via un notebook et s'exécute entièrement en local.
Maintenant ce projet fonctionne via :
- un notebook (version initial)
- un script main.py : requêtes via le terminal
- une API REST : requête POST dans FastAPI

## 🏗️ Architecture du Projet

Le projet sépare strictement la phase d'ingestion de données (Data Engineering) et la phase de génération de réponses (AI Engineering) :

1. **Scraping & Ingestion :** Extraction de la documentation web brute.
2. **Data Lake (Cloud) :** Stockage persistant des textes bruts dans **Supabase** (PostgreSQL).
3. **Vectorisation & Indexation :** Extraction depuis Supabase, segmentation par texte fixe avec chevauchement (*overlap*), et stockage des vecteurs dans **ChromaDB** en local.
4. **Pipeline RAG :** Recherche sémantique *cross-lingual* et génération de réponses via l'API **Gemini 2.5 Flash**.

---

## 📂 Structure des Fichiers

```text
├── .env                   # Variables d'environnement et clés API (ignoré par Git)
├── .gitignore             # Configuration des fichiers à exclure du dépôt
├── RAG_v0.ipynb           # Notebook de prototypage complet du pipeline v1
└── README.md              # Documentation du projet
```


## Instructions d'intallation et d'utilisation

**Prérequis :**

* python 3.10.6

- les packages : supabase, chromadb et google-genai
  - ```
    !pip install supabase
    ```

    ```
    !pip install chromadb
    ```

    ```
    !pip install google-genai

    ```
- un compte **Supabase** pour obtenir une clé api et un url de connexion
- un compte google pour avoir utiliser **aistudio** avec une clé api

**Configuration**

Pour commencer, créez un ficheir ``.env`` à la racine du projet. Ce fichier contient l'ensemble des variables d'environnement nécessaires pour le bon fonctionnement du projet.

```
SUPABASE_URL="url-supabase"
SUPABASE_KEY="your-anon-key"
GEMINI_API_KEY="YourGeminiKey..."
```


# Axes d'améliorations

* Amélioration du nettoyage des docuements
* Utiliser des méthodes de chunking plus adaptés
* Explorer les méthodes de de calcul de distance
* Ajouter une API
* Ajouter de la persistence de mémoire
