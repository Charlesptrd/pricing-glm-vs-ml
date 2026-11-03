Ce projet construit un tarif d'assurance automobile à partir d'un portefeuille réel de 680 000 polices françaises. La prime pure est estimée par la décomposition classique fréquence × sévérité, à l'aide de modèles linéaires généralisés (Poisson et Gamma). Ces modèles de référence sont ensuite confrontés à un modèle de gradient boosting, afin d'évaluer si le gain prédictif justifie la perte d'interprétabilité — contrainte centrale dans un contexte réglementé. Le travail se prolonge par le passage de la prime pure à la prime commerciale, via un chargement de risque inspiré des approches Solvabilité II.



 - sev : retrait des sinistres dont l'IDpol est absent de freq
        (195 lignes, 6 polices, 1,3 % de la charge)