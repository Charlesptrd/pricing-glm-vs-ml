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
        # pour un ajout, on veut que le gain soit supérieur aux seuils
        rejeter_h0_test = lr_stat >= seuil_lr
        rejeter_h0_aic = lr_stat >= seuil_aic
        rejeter_h0_bic = lr_stat >= seuil_bic
        msg_ok = "Rejet de H0 : L'ajout est significatif (Modèle complexe retenu)",
        msg_ko = "Non-rejet de H0 : L'ajout n'est pas significatif (Modèle simple conservé)"

        print(
            f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ {msg_ok if rejeter_h0_test else msg_ko}"
        )
        print(
            f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ {msg_ok if rejeter_h0_aic else msg_ko}"
        )
        print(
            f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ {msg_ok if rejeter_h0_bic else msg_ko}"
        )

    else:  # action == 'RETRAIT_OU_FUSION'
        # On rejette la simplification (H0) si la perte d'information (LR) dépasse le seuil
        rejeter_h0_test = lr_stat > seuil_lr
        rejeter_h0_aic = lr_stat > seuil_aic
        rejeter_h0_bic = lr_stat > seuil_bic

        msg_ok = "❌ Rejet de H0 : La simplification dégrade trop le modèle (Rester sur le modèle complexe)"
        msg_ko = "✅ Non-rejet de H0 : La simplification est acceptable (Modèle simplifié conservé par parcimonie)"

        # Attention ici : le "succès" de la fusion correspond au NON-REJET de H0 (msg_ko)
        print(
            f"1. Test LR (α={alpha*100}%)   : Seuil = {seuil_lr:.2f} ➔ {msg_ko if not rejeter_h0_test else msg_ok}"
        )
        print(
            f"2. Critère AIC        : Seuil = {seuil_aic:.2f} ➔ {msg_ko if not rejeter_h0_aic else msg_ok}"
        )
        print(
            f"3. Critère BIC        : Seuil = {seuil_bic:.2f} ➔ {msg_ko if not rejeter_h0_bic else msg_ok}"
        )


    return {
        "lr_stat": lr_stat,
        "p_value": p_value,
        "ddl": ddl,
        "action": action,
        "valide_test": rejeter_h0_test,
        "valide_aic": rejeter_h0_aic,
        "valide_bic": rejeter_h0_bic}