1. PySift master est le code original du SIFT from scratch. Celui que je n'ai
quasiment pas modifié.

2. si tu run main.py tu vas voir ce que donne le code que j'ai trouvé sur internet avec 
quasiment pas de modifications.

3. Dans le notebook sift_by_hand, il y a quasiment les 3 premières parties du sift. Je voulais 
pour la partie orientation ne pas coder la parabole. Ca ne me parait pas indispensable. On peut faire
une légère simplification.
Je suis sure par contre qu'il y a un problème avec les valeurs de l'orientation et de la valeur du rayon.
Je ne sais pas trop comment faire.

4. Le notebook project.ipynb montre l'implémentation des keypoints avec le harris detector. Ce que je ne comprends
pas c'est que même si on a des keypoints pour les images de taille divisée par 2 à chaque fois, on n'a quand même
pas d'octaves donc je ne sais pas trop comment on doit faire pour la suite.

Bilan:
Je te conseille de commencer par le notebook sift_by_hand pour voir si tu es d'accord avec ce que j'ai fait et 
le finir.
voici les liens qui pourront t'aider:
- https://lerner98.medium.com/implementing-sift-in-python-36c619df7945
- et le lien du github de l'article: https://github.com/SamL98/PySIFT
- le papier sift que j'ai mis dans ressources

Pour la partie matching, il y a les 3 fichiers:
- nearest_neighbors_matching
- Lowe_NN_matching
- geometric_matching
