"""Labels used in trained FastText models for Occitan dialect classification"""
from enum import StrEnum

FASTTEXT_PREFIX = "__label__"


class DidLabels(StrEnum):
    # AGUIAIN = "agui"
    ARAN = "ara"
    AUVERN = "auv"
    # CISAUP = "cis"
    GASCON = "gas"
    LENGADOCIAN = "lan"
    LEMOSIN = "lim"
    # MARCH = "mar"
    NICARD = "nic"
    PROVENC = "pro"
    VIVARO = "viv"


# Dicts to convert labels from diverse sources to the ones used in our FastText models
FASTTEXT_IDENTITY = {lab: lab for lab in DidLabels}

CONGRES_FULL_TO_LABELS = {
    "oc-aranes-grclass": DidLabels.ARAN,
    "oc-auvern-grclass": DidLabels.AUVERN,
    # "oc-cisaup-grclass": DidLabels.CISAUP,
    "oc-gascon-grclass": DidLabels.GASCON,
    "oc-lengadoc-grclass": DidLabels.LENGADOCIAN,
    "oc-lemosin-grclass": DidLabels.LEMOSIN,
    "oc-nicard-grclass": DidLabels.NICARD,
    "oc-nicard-grmistr": DidLabels.NICARD,
    "oc-provenc-grclass": DidLabels.PROVENC,
    "oc-vivaraup-grclass": DidLabels.VIVARO
}

CONGRES_STRIPPED_TO_LABELS = {
    "aranes": DidLabels.ARAN,
    "auvern": DidLabels.AUVERN,
    # "cisaup": DidLabels.CISAUP,
    "gascon": DidLabels.GASCON,
    "lengadoc": DidLabels.LENGADOCIAN,
    "lemosin": DidLabels.LEMOSIN,
    "nicard": DidLabels.NICARD,
    "provenc": DidLabels.PROVENC,
    "vivaraup": DidLabels.VIVARO
}

TTB_TO_LABELS = {
    "gascon": DidLabels.GASCON,
    "languedocien": DidLabels.LENGADOCIAN,
    "limousin": DidLabels.LEMOSIN,
    "provencal": DidLabels.PROVENC
}

WIKI_TO_LABELS = {
    # "aguiainés": DidLabels.AGUIAIN,
    "aranés": DidLabels.ARAN,
    "auvernhat": DidLabels.AUVERN,
    "gascon": DidLabels.GASCON,
    "lemosin": DidLabels.LEMOSIN,
    "lengadocian": DidLabels.LENGADOCIAN,
    # "marchés": DidLabels.MARCH,
    "niçard": DidLabels.NICARD,
    "provençau": DidLabels.PROVENC,
    "vivaroaupenc": DidLabels.VIVARO
}

FORUM_TO_LABELS = {
    "gasc-1240": DidLabels.GASCON,
    "lang-1309": DidLabels.LENGADOCIAN,
    "limo-1246": DidLabels.LEMOSIN,
    "prov-1235": DidLabels.PROVENC
}

LABEL_SCHEME_TO_MAPPER = {
    "congres_full": CONGRES_FULL_TO_LABELS,
    "congres_stripped": CONGRES_STRIPPED_TO_LABELS,
    "ttb": TTB_TO_LABELS,
    "wiki": WIKI_TO_LABELS,
    "forum": FORUM_TO_LABELS,
    "fasttext": FASTTEXT_IDENTITY
}

SOURCE_TO_SCHEME = {
    "wiki": "wiki",
    "ttb": "ttb",
    "congres": "congres_stripped",
    "soft": "congres_stripped",
    "forum": "forum",
    "concat": "fasttext"
}
