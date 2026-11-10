import statsmodels.api as sm
import statsmodels.formula.api as smf

def ajuster_gamma(formule, data):
    return smf.glm(formule, data=data, family=sm.families.Gamma(link=sm.families.links.Log())).fit()