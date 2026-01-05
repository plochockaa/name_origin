# This script creates cleaned datasets of forenames and surnames from two separate databases.
# 
import pandas as pd
import pycountry
from names_dataset import NameDataset
from utils.cleaning import normalize_name
# =====================================================
# CONFIG
# =====================================================

WRITE_CSV = True
OUT_SURNAME_CSV = "data/final_surnames.csv"
OUT_FORENAME_CSV = "data/final_forenames.csv"

forenames_csv = "data/common-forenames-by-country.csv"
surnames_csv  = "data/common-surnames-by-country.csv"

# =====================================================
# HELPERS
# =====================================================


def unique_sorted(series):
    return sorted(set(x for x in series if pd.notna(x)))


def build_final_table(sig_df, nd_df, name_type):
    """
    Build final table:
    one row per name, countries collapsed into a list
    """

    sig = sig_df[sig_df["type"] == name_type].copy()
    nd  = nd_df[nd_df["type"] == name_type].copy()

    joined = sig.merge(
        nd,
        on=["name_norm", "type"],
        how="inner",
        suffixes=("_sigpwned", "_nd"),
        validate="many_to_many"
    )

    # combine country evidence
    joined["countries"] = (
        joined[["country_name_sigpwned", "country_name_nd"]]
        .values
        .tolist()
    )

    final = (
        joined
        .groupby("name_norm", as_index=False)
        .agg({
            "countries": lambda x: sorted(set(c for row in x for c in row if pd.notna(c)))
        })
        .rename(columns={"name_norm": "name"})
    )

    return final

# =====================================================
# 1) LOAD SIGPWNED DATA
# =====================================================

df_sig_forenames = pd.read_csv(forenames_csv)
df_sig_surnames  = pd.read_csv(surnames_csv)

df_sig_forenames["source"] = "sigpwned"
df_sig_forenames["type"]   = "forename"
df_sig_surnames["source"]  = "sigpwned"
df_sig_surnames["type"]    = "surname"

combined = pd.concat([df_sig_forenames, df_sig_surnames], ignore_index=True)

combined = (
    combined
    .rename(columns={
        "Localized Name": "original_name",
        "Romanized Name": "name"
    })[["Country", "original_name", "name", "source", "type"]]
    .dropna(subset=["name"])
)

# country lookup
country_lookup = {c.alpha_2: c.name for c in pycountry.countries}
combined["country_name"] = combined["Country"].map(country_lookup)
combined.drop(columns="Country", inplace=True)

combined["name_norm"] = combined["name"].apply(normalize_name)

# =====================================================
# 2) LOAD NAMES_DATASET
# =====================================================

nd = NameDataset()
records = []

sample_names = (
    combined["name"]
    .dropna()
    .unique()
    .tolist()
)

print(len(sample_names), "unique names to look up in names_dataset")

for nm in sample_names:
    out = nd.search(nm)

    if out.get("first_name"):
        records.append({
            "name": nm,
            "type": "forename",
            "source": "names_dataset",
            "countries": list(out["first_name"]["country"].keys())
        })

    if out.get("last_name"):
        records.append({
            "name": nm,
            "type": "surname",
            "source": "names_dataset",
            "countries": list(out["last_name"]["country"].keys())
        })

df_nd = pd.DataFrame(records)
df_nd["name_norm"] = df_nd["name"].apply(normalize_name)

df_nd = (
    df_nd
    .explode("countries")
    .rename(columns={"countries": "country_code"})
    .assign(country_name=lambda x: x["country_code"].map(country_lookup))
)

# =====================================================
# 3) BUILD FINAL TABLES
# =====================================================

final_surnames  = build_final_table(combined, df_nd, "surname")
final_forenames = build_final_table(combined, df_nd, "forename")

print("SURNAMES")
print(final_surnames.head())

print("\nFORENAMES")
print(final_forenames.head())

# =====================================================
# 4) WRITE CSVs (OPTIONAL)
# =====================================================

if WRITE_CSV:
    final_surnames.to_csv(OUT_SURNAME_CSV, index=False)
    final_forenames.to_csv(OUT_FORENAME_CSV, index=False)