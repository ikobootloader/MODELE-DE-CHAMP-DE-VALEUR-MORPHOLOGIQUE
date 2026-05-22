# Champ de Valeur Morphologique (CVM) pour la reconnaissance de formes

Ce projet propose une reformulation de la propagation de récompense en un modèle de reconnaissance interprétable, appelé **Champ de Valeur Morphologique** (CVM), ou **Morphological Value Field** (MVF).

L'idée centrale: une lettre n'est pas seulement un tracé binaire, mais un **champ de compatibilité spatiale** appris à partir d'exemples.

## 1. Positionnement conceptuel

Le prototype initial utilisait un vocabulaire de renforcement (récompense, propagation, facteur gamma). Le comportement observé correspond plus précisément à:

- une diffusion de valeur sur grille,
- un champ de potentiel discret,
- une empreinte morphologique tolérante.

Le CVM conserve la logique de valeur, sans surpromesse de RL complet.

## 2. Trois matrices distinctes

Pour chaque classe de lettre L:

1. Matrice d'observation (entrée):

   I(x, y)

   binaire (0/1) ou intensité normalisée.

2. Matrice de fréquence (apprentissage):

   F_L(x, y)

   fréquence d'activation du pixel (x, y) dans les exemples de L.

3. Matrice de valeur morphologique (diffusion):

   V_L(x, y) = Σ_{u,v} F_L(u, v) * gamma^(|x-u| + |y-v|)

Le terme Manhattan |x-u| + |y-v| encode la proximité spatiale.

## 3. Fonction de reconnaissance

Score positif (compatibilité):

Score(I, L) = Σ_{x,y} I(x, y) * V_L(x, y)

Pénalisation optionnelle des zones incompatibles:

Score(I, L) = Σ I*V_L - lambda * Σ I*P_L

avec P_L(x, y) = 1 - F_L(x, y) dans la version simple.

Décision par compétition inter-classes:

L_hat = argmax_L Score(I, L)

## 4. Fonctions de score

Le script supporte trois scores:

- `raw`: produit scalaire pénalisé brut,
- `mass`: normalisation par masse `(<I,V> - λ<I,P>) / ((ΣI)(ΣV))`,
- `cosine` (recommandé): similarité cosinus pénalisée.

Par défaut, l'évaluation CVM utilise `cosine`.

## 5. Interprétation de gamma

- gamma élevé: grande tolérance aux variations.
- gamma faible: modèle plus strict, proche du pixel exact.

Extensions possibles:

- gamma global (actuel),
- gamma par classe gamma_L,
- gamma local gamma_L(x, y).

## 6. Ce que fait le script `rec_lettre.py`

Le script fournit un prototype testable avec:

- calcul de `F_L` via moyenne d'exemples,
- diffusion `F_L -> V_L` (diffusion Manhattan séparée rapide),
- construction d'une pénalité simple `P_L`,
- prétraitement (binarisation, recadrage, recentrage, redimensionnement),
- score modulo transformations affines restreintes,
- contraintes topologiques (composantes, trous, extrémités, jonctions),
- double champ (`pixel` + `squelette`) et combinaison pondérée,
- tuning simple sur mini validation,
- baselines: template cosine brut, template recentré, distance transform, chamfer-like.

## 7. Lien avec des familles connues

Le CVM est proche de:

- template matching,
- transformée de distance / chamfer,
- champs de potentiel,
- noyaux de diffusion spatiale (style KDE discret).

Sa spécificité est l'interprétation unifiée en **champ de valeur morphologique appris**.

## 8. Feuille de route expérimentale

1. Prototype binaire minimal (fait).
2. Plusieurs exemples par classe (fait, version jouet).
3. Tests sur classes confusables (O/C/Q, I/L/T, A/H, P/R/B, U/V/Y).
4. Étude de sensibilité à gamma.
5. Ajout et calibrage des pénalités lambda.
6. Passage à des jeux réels (MNIST puis EMNIST).

## 9. Hypothèse de travail

Une forme manuscrite peut être reconnue par projection sur des champs de valeur morphologique appris. La diffusion spatiale contrôlée par gamma intègre les variations locales, tout en conservant une lecture interprétable et un coût de calcul modéré.

## 10. Résultats démo (jeu jouet)

Mesures issues de `run_demo(show_plot=False)` sur le mini jeu interne (6 échantillons de validation):

| Méthode | Accuracy | Temps |
|---|---:|---:|
| CVM base | 0.667 | 167.1 ms |
| CVM enrichi (tuning restreint) | 0.833 | 1485.1 ms |
| Baseline template cosine (raw) | 1.000 | 0.2 ms |
| Baseline template cosine (centered) | 0.833 | 34.9 ms |
| Baseline distance transform | 0.500 | 27.3 ms |
| Baseline chamfer-like | 0.833 | 31.1 ms |

Ces résultats sont purement démonstratifs (jeu minuscule), pas des conclusions scientifiques.

## 11. Validation croisée et premier benchmark hors jouet

Ajouts récents:

- validation croisée `k=2` sur le jeu jouet interne;
- premier benchmark sur `sklearn.datasets.load_digits` (10 classes, binarisation simple).

Mesures observées:

- CVM toy cross-val mean accuracy (`k=2`): `0.625` (folds: `[0.5, 0.75]`)
- CVM digits benchmark (`test=300`): `0.407`
- k-NN pixels (`k=3`, même split digits): `0.950`
- régression logistique pixels (même split digits): `0.957`
- SVM RBF pixels (même split digits): `0.987`

Ablation CVM sur le jeu jouet (split de démo):

- champ seul: `0.833`
- champ + pénalité: `0.833`
- champ + pénalité + transformations: `0.833`
- champ + pénalité + transformations + topologie: `0.833`

Ablation CVM sur un set de stress (rotations faibles + bruit + ruptures):

- champ seul: `0.667`
- champ + pénalité: `0.667`
- champ + pénalité + transformations: `0.625`
- champ + pénalité + transformations + topologie: `0.667`

Sélection de transformations (set de stress):

- score identique observé (`0.583`) entre recherche complète et sélection heuristique;
- coût mesuré quasi identique dans l'état actuel (`~8225 ms` vs `~8140 ms`), donc optimisation encore insuffisante.

Lecture: le modèle reste cohérent comme preuve de concept interprétable, mais les performances montrent qu'il faut encore optimiser le protocole et les hyperparamètres avant toute comparaison sérieuse à des méthodes plus standard.

## 12. Résultats préliminaires (lecture honnête)

Séparation du coût CVM enrichi (run actuel):

- tuning: `791.8 ms`
- inférence (paramètres fixés): `123.9 ms`

Protocole à 3 niveaux:

- `propre`: CVM `0.833`, template centered `0.833`, chamfer `0.833`
- `perturbé`: CVM `0.542`, template centered `0.667`, chamfer `0.583`
- `ambigu`: CVM `0.625`, template centered `0.750`, chamfer `0.625`

Conclusion provisoire:

- sur données propres/alignées, le CVM ne dépasse pas les baselines simples;
- sur les perturbations simulées actuelles, le CVM reste en retrait;
- l'intérêt principal du CVM reste aujourd'hui l'interprétabilité et le cadre méthodologique, plus qu'un gain de performance brute.

## 13. Volet interprétabilité

Le script inclut un outil d'explication locale de prédiction:

- fonction: `explain_prediction(...)` dans `rec_lettre.py`;
- visualisations produites: image d'entrée, champ de la classe vraie, champ de la classe prédite, cartes d'activation `I*V`, cartes de pénalité `I*P`;
- usage: inspecter un exemple mal classé pour comprendre spatialement l'erreur (zones activées vs zones pénalisées).

Étude gamma sur split disjoint `sklearn digits` (train/test séparés):

- `gamma=0.50` -> `0.767`
- `gamma=0.60` -> `0.740`
- `gamma=0.70` -> `0.717`
- `gamma=0.80` -> `0.697`
- `gamma=0.90` -> `0.627`
- `gamma=0.95` -> `0.520`

Lecture: sur ce benchmark, une diffusion trop large dégrade la discrimination; le meilleur régime observé est à gamma plus faible.

Étude complémentaire `train/val/test` + point no-diffusion (split 3 voies):

- `gamma=0.10`: val `0.775`, test `0.775`
- `gamma=0.20`: val `0.770`, test `0.775`
- `gamma=0.30`: val `0.765`, test `0.750`
- `gamma=0.40`: val `0.760`, test `0.740`
- `gamma=0.50`: val `0.755`, test `0.735`
- `no-diffusion (V_L = F_L)`: val `0.765`, test `0.790`

Lecture prudente: le meilleur gamma sur validation est à la borne basse testée (`0.10`) et le point no-diffusion surpasse ce choix en test. À ce stade, la diffusion exponentielle ne montre pas de bénéfice clair sur ce benchmark; elle peut même nuire à la discrimination.

Test spatial complémentaire (digits décalés aléatoirement, non recentrés, split 3 voies, évaluation sans recentrage):

- `gamma=0.10`: val `0.255`, test `0.215`
- `gamma=0.20`: val `0.200`, test `0.175`
- `gamma=0.30`: val `0.155`, test `0.145`
- `gamma=0.40`: val `0.145`, test `0.090`
- `gamma=0.50`: val `0.140`, test `0.075`
- `no-diffusion`: val `0.265`, test `0.225`

Lecture: même dans un régime avec variabilité spatiale artificielle, la diffusion testée ici ne montre pas de gain; le point no-diffusion reste meilleur sur validation et test.

Version corrigée (petits décalages ±1/±2, translations explicites désactivées, sans recentrage):

- `gamma=0.30`: val `0.280`, test `0.240`
- `gamma=0.50`: val `0.200`, test `0.140`
- `gamma=0.70`: val `0.175`, test `0.125`
- `no-diffusion`: val `0.405`, test `0.255`
- hasard (10 classes): `0.100`

Garde-fou d'interprétabilité: `feasible_above_chance = False` (toutes les variantes restent loin d'un régime de performance exploitable).

Conclusion prudente: ce test spatial corrigé reste non concluant pour juger finement diffusion vs no-diffusion, car le niveau absolu de performance est trop bas.
