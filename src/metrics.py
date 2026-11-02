import numpy as np
import pandas as pd
from scipy import stats

def table_frequence(df, col):
    t = df.groupby(col, observed=True).agg(exposition=("Exposure", "sum"), sinistres=("ClaimNb", "sum"),)
    t["frequence"] = t["sinistres"] / t["exposition"]
    t["sigma"] = np.sqrt(t["sinistres"]) / t["exposition"]
    return t

def comparer_tranches(df, col):
    t = table_frequence(df, col)

    f, s = t["frequence"].to_numpy(), t["sigma"].to_numpy()
    noms = t.index.to_numpy()

    ecart = np.abs(np.diff(f))
    sigma_ecart = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)

    res = pd.DataFrame({
        "paire": [f"{a} / {b}" for a, b in zip(noms[:-1], noms[1:])],
        "ecart": ecart,
        "sigma_ecart": sigma_ecart,
        "rapport": ecart / sigma_ecart,
    })

    return res

def seuil_test_lr(ddl, alpha=0.05):
    return stats.chi2.ppf(1-alpha, df=ddl).round(2)
def seuil_aic_lr(ddl):
    return 2.0*ddl

def seuil_bic_lr(ddl, n):
    return (np.log(n)*ddl).round(2)

def comparer_modeles_glm(modele_ref, modele_nouveau, alpha = 0.05):


    if modele_nouveau.df_model > modele_ref.df_model:
        modele_simple, modele_complexe = modele_ref, modele_nouveau
        action = "AJOUT"
    else:
        modele_simple, modele_complexe = modele_nouveau, modele_ref
        action = "RETRAIT OU FUSION"


    ddl = modele_complexe.df_model - modele_simple.df_model

    if modele_ref.nobs != modele_nouveau.nobs:
            raise ValueError("les deux modèles ne portent pas sur le même échantillon")
    if ddl == 0:
        raise ValueError("modèles de même dimension : rien à comparer")

    
    lr_stat = float(2 * (modele_complexe.llf - modele_simple.llf))
    p_value = float(stats.chi2.sf(lr_stat, df=ddl))


    seuil_lr = seuil_test_lr(ddl, alpha=alpha)
    seuil_aic = seuil_aic_lr(ddl)
    seuil_bic = seuil_bic_lr(ddl, modele_ref.nobs)

    print(f"=== COMPARAISON DE MODÈLES GLM (n = {modele_ref.nobs}, |Δddl| = {ddl}) ===")
    print(f"Action détectée : {action}")
    print(f"Statistique LR calculée : {lr_stat:.2f} (p-value : {p_value:.4e})")
    print("-" * 65)

    if action == "AJOUT":
        # On accepte la complexification (H0) si le gain le gain d'information est supérieur aux seuils
        rejeter_h0_test = lr_stat >= seuil_lr
        rejeter_h0_aic = lr_stat >= seuil_aic
        rejeter_h0_bic = lr_stat >= seuil_bic

        if rejeter_h0_test:
            print(f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ OK Rejet de H0 : Les données permettent de mettre en évidence que l'ajout de ce/ces coefficient(s) améliorent significativement le modèle.")
        else:
            print(f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ KO Non-rejet de H0 : Les donnnées ne permettent pas de mettre en évidence une amélioration significative du modèle par l'ajout de nouveaux coefficients.")
        if rejeter_h0_aic:
            print(f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ OK Changement validé pour l'AIC.")
        else:
            print(f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ KO Changement rejeté pour l'AIC.")
        if rejeter_h0_bic:
            print(f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ OK Changement validé pour le BIC.")
        else:
            print(f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ KO Changement rejeté pour le BIC.")



    else:  
        # On rejette la simplification (H0) si la perte d'information (LR) dépasse le seuil
        rejeter_h0_test = lr_stat > seuil_lr
        rejeter_h0_aic = lr_stat > seuil_aic
        rejeter_h0_bic = lr_stat > seuil_bic

        # Attention le succès de la fusion correspond au NON-REJET de H0 (msg_ko)

        if rejeter_h0_test:
            print(f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ KO Non-rejet de H0 : Les données ne permettent pas de mettre en évidence une amélioration significative du modèle.")
        else:
            print(f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ OK Rejet de H0 : Les données permettent de mettre en évidence que la simplification améliore significativement le modèle.")
        if rejeter_h0_aic:
            print(f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ KO Changement rejeté pour l'AIC.")
        else:
            print(f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ OK Changement validé pour l'AIC.")
        if rejeter_h0_bic:
            print(f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ KO Changement rejeté pour le BIC.")
        else:
            print(f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ OK Changement validé pour le BIC.")

    if action == "AJOUT":
        return {
            "lr_stat": lr_stat,
            "p_value": p_value,
            "ddl": ddl,
            "action": action,
            "valide_test": rejeter_h0_test,
            "valide_aic": rejeter_h0_aic,
            "valide_bic": rejeter_h0_bic}
    else:
        return {
                    "lr_stat": lr_stat,
                    "p_value": p_value,
                    "ddl": ddl,
                    "action": action,
                    "valide_test": not rejeter_h0_test,
                    "valide_aic": not rejeter_h0_aic,
                    "valide_bic": not rejeter_h0_bic}




def lift_table(modele, data:pd.DataFrame, n_groupes=10, recalage=1.0):

    d = data.copy()
    offset = np.log(d["Exposure"])
    d["mu"] = modele.predict(d, offset=offset) * recalage

    #Calcul des fréquences annualisées prédites 
    d["prime"] = d["mu"] / d["Exposure"]  


    d = d.sort_values("prime") # On les classe de la moins risquée à la plus risquée.
    cum = d["Exposure"].cumsum() / d["Exposure"].sum() # calcul pour chaque police de la part d'exposition située avant elle

    d["groupe"] = np.minimum((cum * n_groupes).astype(int), n_groupes - 1) # chaque police recoit un numéro de 1 à 9 selon sa part d'exposition situé avant elle

    t = d.groupby("groupe").agg(exposition=("Exposure", "sum"),sinistres=("ClaimNb", "sum"),sinistres_pred=("mu", "sum")) # on regroupe par numéro et on ajoute les variables intéressantes


    t["observe"] = t["sinistres"] / t["exposition"] # On calcule la fréquence empirique sur le groupe
    t["predit"] = t["sinistres_pred"] / t["exposition"] # On calcule la fréquence prédite sur le groupe.
    t["sigma"] = np.sqrt(t["sinistres"]) / t["exposition"] # Calcule de l'écart type.
    return t

def gini(modele, data, recalage=1.0):
    d = data.copy()
    offset = np.log(d["Exposure"])
    d["mu"] = modele.predict(d, offset=offset) * recalage
    d["prime"] = d["mu"] / d["Exposure"]

    d = d.sort_values("prime")

# jusque là, même chose que pour le Lift-Chart

    x = np.concatenate([[0], d["Exposure"].cumsum() / d["Exposure"].sum()]) # diagonale
    y = np.concatenate([[0], d["ClaimNb"].cumsum() / d["ClaimNb"].sum()]) # courbe 

    gini = 1 - 2 * np.trapezoid(y, x) # on intègre y selon la courbe x (la diagonale) selon la méthode des trapèzes.
    return x, y, gini