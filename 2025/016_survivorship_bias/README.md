# Le Biais du Survivant en Crypto Trading

Ce projet illustre le **biais du survivant** dans l'analyse des cryptomonnaies en téléchargeant et visualisant l'historique complet du Top 30 depuis 2020.

## 🎯 Objectif

Démontrer pourquoi analyser uniquement les cryptos actuellement dans le Top 30 crée un biais dangereux pour le trading algorithmique.

## 📹 Vidéo YouTube

Lien vers la vidéo explicative : [À venir]

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Cloner le repository**
```bash
cd survivorship_bias
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Lancer Jupyter**
```bash
jupyter notebook survivorship_bias.ipynb
```

## 📊 Contenu du Notebook

### 1. Téléchargement des données
- Utilise l'API publique de CoinMarketCap
- Récupère le Top 30 mensuellement depuis 2020-01-01
- Exclut les stablecoins et wrapped tokens
- **Aucune clé API requise**

### 2. Visualisations

#### Graphique Principal : Timeline du Top 30
- **Axe X** : Temps (mois de 2020 à 2025)
- **Axe Y** : Tous les symboles ayant été dans le Top 30
- **Couleur** :
  - ⚪ **Blanc** : Actif présent dans le Top 30
  - 🔴 **Rouge** : Actif sorti du Top 30

#### Graphiques Complémentaires
- Durée de présence de chaque actif
- Comparaison Top 30 : 2020 vs 2025
- Statistiques de turnover

### 3. Statistiques Clés
- Nombre total d'actifs passés dans le Top 30
- Taux de survivants vs disparus
- Actifs notables disparus (LUNA, FTT, etc.)
- Impact du biais sur les backtests

## 💡 Concept : Le Biais du Survivant

### L'Histoire des Avions (WWII)
Durant la Seconde Guerre mondiale, l'armée américaine voulait renforcer ses avions. En analysant les impacts de balles sur les avions qui revenaient, ils pensaient renforcer les zones touchées.

**Abraham Wald** a révélé l'erreur : il fallait renforcer les zones **sans impacts**, car les avions touchés à ces endroits ne revenaient jamais.

### Application au Trading Crypto

**Erreur classique** :
- Analyser uniquement les cryptos actuellement dans le Top 30
- Ignorer les projets disparus (LUNA, FTT, BitConnect, etc.)
- Surestimer les performances réelles

**Notre approche** :
- Suivre le Top 30 **mois par mois**
- Inclure tous les actifs (survivants + disparus)
- Capturer le risque réel de "mort" d'un actif

## 📈 Résultats Attendus

Le notebook révèle :
- ~40-50 actifs différents sont passés dans le Top 30 depuis 2020
- ~30-40% de turnover (actifs disparus)
- Les backtests sans biais du survivant donnent des résultats très différents

## 🔧 Configuration

Vous pouvez modifier les paramètres dans le notebook :

```python
START_DATE = '2020-01-01'  # Date de début
END_DATE = '2025-11-01'    # Date de fin
TOP_N = 30                 # Taille du classement
EXCLUDE_TAGS = ['stablecoin', 'wrapped-tokens']  # Tags à exclure
```

## 📦 Structure des Fichiers

```
survivorship_bias/
│
├── survivorship_bias.ipynb       # Notebook principal
├── requirements.txt              # Dépendances Python
├── README.md                     # Ce fichier
│
└── [Générés après exécution]
    ├── top30_historical_data.csv
    ├── presence_matrix.csv
    ├── survivorship_bias_timeline.png
    ├── duration_top30.png
    └── comparison_start_end.png
```

## ⚠️ Limitations

- **Taux de requêtes** : L'API gratuite de CoinMarketCap limite à ~333 appels/jour
- **Données historiques** : Limitées à la disponibilité de CoinMarketCap
- **Pause entre requêtes** : 0.5s pour éviter de surcharger l'API

## 🤝 Contribution

Ce projet est open-source. N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Partager vos résultats

## 📚 Pour Aller Plus Loin

### Trading Algorithmique Sans Biais
Pour éviter le biais du survivant dans vos backtests :

1. **Univers dynamique** : Utilisez le Top N à chaque période (pas rétrospectivement)
2. **Inclure les morts** : Gardez les actifs delistés/disparus dans l'historique
3. **Simulation réaliste** : Gérez les entrées/sorties du classement

### Ressources
- [Article original d'Abraham Wald](https://en.wikipedia.org/wiki/Survivorship_bias)
- [CoinMarketCap API Documentation](https://coinmarketcap.com/api/)

## 📧 Contact

Questions ? Suggestions ? Laissez un commentaire sur la vidéo YouTube !

---

**⭐ Si ce projet vous aide, laissez une étoile et abonnez-vous à la chaîne !**
