# Prédire les « Bad Buzz » grâce à l’IA : une approche MLOps complète

---

*Ce billet explique étape par étape comment nous avons développé un modèle IA capable d’anticiper les sentiments négatifs sur Twitter, dans un contexte professionnel réel avec la compagnie Air Paradis.*

---

## **1. Contexte et enjeux**

Air Paradis, compagnie aérienne, souhaite anticiper les crises de réputation sur les réseaux sociaux, notamment Twitter.

Objectifs du projet :

- Développer un prototype d’IA pour prédire les sentiments des tweets.
- Intégrer une démarche MLOps (Machine Learning Operations).
- Limiter les coûts de déploiement grâce à des outils cloud gratuits.

![image.png](assets/image.png)

*(crédit: Ubaid Shah)*

---

## **2. Méthodologie : Comparaison de plusieurs modèles**

Pour répondre efficacement au besoin d’Air Paradis, nous avons testé plusieurs approches de modélisation :

### **Approche 1 : Modèle classique (simple)**

- Modèles testés :
    - **Régression logistique**.
    - **Ridge Classifier**
    - **Light Gradient Boosting Machine**
    - **Gradient Boosting Classifier**
    - **RandomForest**
- Embeddings utilisés :
    - **TF-IDF** (Vectorisation basique des mots).
    - **Word2Vec**
    - **FastText**
- Modél retenu: Logistic regression  + word2vec.

![image.png](assets/image%201.png)

### 🧠 **Approche 2 : Modèle avancé (Deep Learning)**

- Modèle testé : **Réseau de neurones profonds** (TensorFlow-Keras).
- Embeddings testés : **Word2Vec**, **FastText**.
- Embedding retenu (meilleures performances se rapprochant de word2vec mais avec un temps d’entrainnement beaucoup plus rapide) : **FastText**.

Resultat d’entrainement avec word2vec:

![image.png](assets/image%202.png)

Resultat d’entrainement avec fasttext:

![image.png](assets/image%203.png)

Comparatif des deux embeddings: **wordvec** vs **fasttext**

![image.png](assets/image%204.png)

### **Approche 3 : Modèle BERT (transformer avancé)**

- Objectif : évaluer si l’investissement dans BERT améliore significativement les résultats.

![image.png](assets/image%205.png)

La matrice de confusion avec bert nous donne ceci:

![image.png](assets/image%206.png)

---

## **3. Intégration du MLOps dans le projet**

La démarche MLOps a permis de structurer, automatiser et optimiser notre cycle de vie des modèles :

### **Tracking et gestion des expériences avec MLflow**

MLflow a permis de :

- Suivre et centraliser les expérimentations.
- Stocker les modèles et leurs versions.
- Tester rapidement les modèles en production via le serving MLflow.

![image.png](assets/image%207.png)

### **Automatisation du déploiement (CI/CD)**

Nous avons utilisé une chaîne complète d’intégration et déploiement continus :

- **Versioning du code :** Git/GitHub.
- **Tests unitaires automatisés :** GitHub Actions.
- **Déploiement :**
    - Hetzner avec un cluster kubernetes (pour reduire les couts de deploiement)
    - Azure Web Apps (ASP F1 gratuit) pour le monitoring avec azure insight

Deploiement du stack (mlflow, minio et postgres)

![image.png](assets/image%208.png)

---

Deploiement du stack (api et front streamlit)

![image.png](assets/image%209.png)

## **4. Monitoring et amélioration continue**

Pour garantir l’efficacité en production, nous avons mis en place :

- **Monitoring avec Azure Application Insights :**
    - Traces des tweets mal prédits.
    - Alertes automatiques par SMS ou email en cas d’anomalie répétée.
- **Démarche d’amélioration continue :**
    - Analyse régulière des erreurs.
    - Réentraînement périodique basé sur les données collectées.

Exemple d’une alerte levée:

![image.png](assets/image%2010.png)

---

## **5. Optimisation et déploiement économique**

Afin de respecter les contraintes budgétaires, nous avons optimisé le modèle :

- Conversion du modèle TensorFlow-Keras en **TensorFlow Lite**.
- Monitoring gratuit grâce à une webapp Azure gratuite.
- Hebergement de mlflow sur une cluster moins couteux chez Hetzner

---

## **6. Résultats et perspectives**

Le modèle avancé (FastText + Deep Learning optimisé) a permis à Air Paradis de disposer d’un outil fiable, performant et économiquement viable pour anticiper les bad buzz.

Grâce à l’intégration complète du MLOps, le produit est prêt à évoluer facilement en fonction des retours terrain.

![image.png](assets/image%2011.png)

![image.png](assets/image%2012.png)

---

## **Conclusion : l’intérêt crucial du MLOps**

Ce projet démontre l’importance essentielle du MLOps pour assurer :

- **Qualité :** tests continus et intégration rigoureuse.
- **Agilité :** adaptation rapide aux besoins réels des utilisateurs.
- **Optimisation économique :** respect des contraintes financières grâce à l’automatisation et l’optimisation technique.

Vous souhaitez implémenter une démarche similaire dans votre entreprise ?

N’hésitez pas à nous contacter pour un accompagnement personnalisé.

---

**Points-clés à retenir :**

- Le MLOps est incontournable pour réussir durablement un projet IA.
- La comparaison rigoureuse des modèles permet une optimisation efficace.
- Des outils gratuits peuvent suffire à déployer efficacement un modèle performant.

---

**Rédigé par : Amadou SY**

---