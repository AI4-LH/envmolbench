"""
内置数据集注册表。

记录所有内置数据集的名称（key）和对应的 CSV 文件名（value）。
路径解析时以 config/base.yaml 中的 datasets_dir 为根目录。

用法示例：
    from envmolbench.data import list_datasets
    print(list_datasets())          # 查看所有数据集名称
    smiles, labels, task = load_dataset("esol")   # 按名称加载
"""

# 注册表：{显示名称: CSV文件名（不含路径）}
# key 用于 API 调用，value 用于文件查找

# web: www.ai4env.world/envmolbench/

# github: https://github.com/AI4-LH/envmolbench

DATASET_REGISTRY = {
    # ── 分类数据集（毒性、活性）──────────────────────────────
    "hepatotoxicity":           "hepatotoxicity.csv",
    "androgen_receptor":         "Androgen Receptor.csv",
    "antibacterial_activity":    "Antibacterial activity.csv",
    "estrogen_receptor":         "Estrogen Receptor α.csv",
    "ames_mutagenicity":         "ames mutagenicity.csv",
    "carcinogenicity":           "carcinogenicity.csv",
    "cytotoxicity":              "cytotoxicity.csv",
    "oral_carcinogenicity":      "oral carcinogenicity.csv",
    "neurotoxicity":             "neurotoxicity.csv",
    "ocular_toxicity":           "ocular toxicity.csv",
    "prenatal_development":      "prenatal development.csv",
    "reproductive_toxicity":     "reproductive toxicity.csv",
    "respiratory_toxicity":      "respiratory toxicity.csv",
    "skin_corrosion":            "skin corrosion.csv",
    "Hepatotoxicants":           "Hepatotoxicants.csv",
    "pbt":                       "PBT.csv",
    "tshr_agonist":              "TSHR agonist activity.csv",

    # ── 回归数据集（物化性质）────────────────────────────────
    "esol":                      "esol.csv",
    "freesolv":                  "freesolv.csv",
    "lipophilicity":             "Lipophilicity.csv",
    "solubility":                "solubility.csv",
    "fubrain":                   "fubrain.csv",
    "fup":                       "fup.csv",
    "hlm":                       "HLM.csv",
    "kh":                        "KH.csv",
    "oral_bioavailability":      "oral bioavailability.csv",
    "pka_acidic":                "pKa acidic.csv",
    "clin_fup":                  "clin fup.csv",
    "pfas":                      "PFAS dataset.csv",
    "pparg_ic50":                "PPARγ_IC50.csv",
    "pparg_pkd":                 "PPARγ_pkd.csv",

    # ── 环境化学数据集 ────────────────────────────────────────
    "aero_bio_c":                "aero bio(C).csv",
    "aero_bio_r":                "aero bio(R).csv",
    "tfishbio":                  "tfishbio.csv",
    "koa":                       "KOA.csv",
    "koc":                       "koc.csv",
    "koc2":                      "KOC_2.csv",
    "tbp":                       "TBP.csv",
    "si_oh":                     "SI OH.csv",
    "si_so4":                    "SI SO4.csv",

    # ── 其他分子性质数据集 ────────────────────────────────────
    "ccs_mh":                    "CCS MH.csv",
    "ccs_mna":                   "CCS MNa.csv",
    "fba":                       "FBA.csv",
    "fbc":                       "FBC.csv",
    "plv":                       "PLV.csv",
}


def list_datasets() -> list:
    """返回所有已注册的内置数据集名称列表。"""
    return sorted(DATASET_REGISTRY.keys())
