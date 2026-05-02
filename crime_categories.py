"""
crime_categories.py
────────────────────────────────────────────────
Shared category map used by both city ETL scripts.

Maps each city's raw crime labels to one standardized
category so both SQL tables use the same vocabulary.
────────────────────────────────────────────────
"""

CRIME_CATEGORY_MAP = {

    # ── Seattle (NIBRS labels — exact strings from raw data) ───────

    # Theft
    "THEFT FROM MOTOR VEHICLE":                         "Theft",
    "ALL OTHER LARCENY":                                "Theft",
    "SHOPLIFTING":                                      "Theft",
    "THEFT OF MOTOR VEHICLE PARTS OR ACCESSORIES":      "Theft",
    "THEFT FROM BUILDING":                              "Theft",
    "POCKET-PICKING":                                   "Theft",
    "PURSE-SNATCHING":                                  "Theft",
    "THEFT FROM COIN-OPERATED MACHINE OR DEVICE":       "Theft",
    "THEFT NEC":                                        "Theft",
    "LARCENY-THEFT":                                    "Theft",
    "STOLEN PROPERTY OFFENSES":                         "Theft",

    # Burglary
    "BURGLARY/BREAKING & ENTERING":                     "Burglary",
    "BURGLARY/BREAKING&ENTERING":                       "Burglary",
    "BURGLARY":                                         "Burglary",

    # Vehicle Theft
    "MOTOR VEHICLE THEFT":                              "Vehicle Theft",

    # Assault
    "SIMPLE ASSAULT":                                   "Assault",
    "AGGRAVATED ASSAULT":                               "Assault",
    "INTIMIDATION":                                     "Assault",
    "ASSAULT OFFENSES":                                 "Assault",

    # Robbery
    "ROBBERY":                                          "Robbery",
    "EXTORTION/BLACKMAIL":                              "Robbery",

    # Homicide
    "MURDER & NONNEGLIGENT MANSLAUGHTER":               "Homicide",
    "NEGLIGENT MANSLAUGHTER":                           "Homicide",
    "HOMICIDE OFFENSES":                                "Homicide",
    "JUSTIFIABLE HOMICIDE":                             "Homicide",

    # Drugs
    "DRUG/NARCOTIC VIOLATIONS":                         "Drugs",
    "DRUG EQUIPMENT VIOLATIONS":                        "Drugs",
    "DRIVING UNDER THE INFLUENCE":                      "Drugs",

    # Vandalism
    "DESTRUCTION/DAMAGE/VANDALISM OF PROPERTY":         "Vandalism",
    "VANDALISM":                                        "Vandalism",

    # Fraud
    "CREDIT CARD/AUTOMATED TELLER MACHINE FRAUD":       "Fraud",
    "IDENTITY THEFT":                                   "Fraud",
    "FALSE PRETENSES/SWINDLE/CONFIDENCE GAME":          "Fraud",
    "IMPERSONATION":                                    "Fraud",
    "WIRE FRAUD":                                       "Fraud",
    "BAD CHECKS":                                       "Fraud",
    "COUNTERFEITING/FORGERY":                           "Fraud",
    "EMBEZZLEMENT":                                     "Fraud",
    "FRAUD OFFENSES":                                   "Fraud",

    # Weapons
    "WEAPON LAW VIOLATIONS":                            "Weapons",

    # Arson
    "ARSON":                                            "Arson",

    # Sex Offenses
    "RAPE":                                             "Sex Offenses",
    "SODOMY":                                           "Sex Offenses",
    "SEXUAL ASSAULT WITH AN OBJECT":                    "Sex Offenses",
    "FONDLING":                                         "Sex Offenses",
    "INCEST":                                           "Sex Offenses",
    "STATUTORY RAPE":                                   "Sex Offenses",
    "SEX OFFENSES":                                     "Sex Offenses",
    "SEX OFFENSES, NONFORCIBLE":                        "Sex Offenses",
    "PORNOGRAPHY/OBSCENE MATERIAL":                     "Sex Offenses",
    "PEEPING TOM":                                      "Sex Offenses",

    # Kidnapping
    "KIDNAPPING/ABDUCTION":                             "Kidnapping",

    # Human Trafficking
    "HUMAN TRAFFICKING, COMMERCIAL SEX ACTS":           "Human Trafficking",
    "HUMAN TRAFFICKING, INVOLUNTARY SERVITUDE":         "Human Trafficking",

    # Prostitution
    "PROSTITUTION OFFENSES":                            "Prostitution",

    # Trespassing
    "TRESPASS OF REAL PROPERTY":                        "Trespassing",
    "TRESPASS OF REAL":                                 "Trespassing",

    # Public Disorder
    "DISORDERLY CONDUCT":                               "Public Disorder",
    "DRUNKENNESS":                                      "Public Disorder",
    "CURFEW/LOITERING/VAGRANCY VIOLATIONS":             "Public Disorder",
    "LIQUOR LAW VIOLATIONS":                            "Public Disorder",

    # Gambling
    "GAMBLING OFFENSES":                                "Gambling",
    "BETTING/WAGERING":                                 "Gambling",

    # Domestic / Family
    "FAMILY OFFENSES, NONVIOLENT":                      "Domestic / Family",

    # Public Corruption
    "BRIBERY":                                          "Public Corruption",

    # Other / non-crime
    "ALL OTHER OFFENSES":                               "Other",
    "NOT REPORTABLE TO NIBRS":                          "Other",
    "ANIMAL CRUELTY":                                   "Other",


    # ── Chicago (Primary Type labels) ──────────────────────────────

    # Theft
    "THEFT":                                            "Theft",

    # Burglary
    "BURGLARY":                                         "Burglary",

    # Vehicle Theft
    "MOTOR VEHICLE THEFT":                              "Vehicle Theft",

    # Assault
    "ASSAULT":                                          "Assault",
    "BATTERY":                                          "Assault",
    "AGGRAVATED ASSAULT":                               "Assault",
    "AGGRAVATED BATTERY":                               "Assault",
    "INTIMIDATION":                                     "Assault",
    "STALKING":                                         "Assault",

    # Robbery
    "ROBBERY":                                          "Robbery",

    # Homicide
    "HOMICIDE":                                         "Homicide",

    # Drugs
    "NARCOTICS":                                        "Drugs",
    "OTHER NARCOTIC VIOLATION":                         "Drugs",

    # Vandalism
    "CRIMINAL DAMAGE":                                  "Vandalism",

    # Fraud
    "DECEPTIVE PRACTICE":                               "Fraud",

    # Trespassing
    "CRIMINAL TRESPASS":                                "Trespassing",

    # Weapons
    "WEAPONS VIOLATION":                                "Weapons",
    "CONCEALED CARRY LICENSE VIOLATION":                "Weapons",

    # Sex Offenses
    "SEX OFFENSE":                                      "Sex Offenses",
    "CRIM SEXUAL ASSAULT":                              "Sex Offenses",
    "CRIMINAL SEXUAL ASSAULT":                          "Sex Offenses",
    "OBSCENITY":                                        "Sex Offenses",

    # Arson
    "ARSON":                                            "Arson",

    # Kidnapping
    "KIDNAPPING":                                       "Kidnapping",

    # Human Trafficking
    "HUMAN TRAFFICKING":                                "Human Trafficking",

    # Prostitution
    "PROSTITUTION":                                     "Prostitution",

    # Public Disorder
    "LIQUOR LAW VIOLATION":                             "Public Disorder",
    "PUBLIC PEACE VIOLATION":                           "Public Disorder",

    # Gambling
    "GAMBLING":                                         "Gambling",

    # Offenses Involving Minors
    "OFFENSE INVOLVING CHILDREN":                       "Offenses Involving Minors",

    # Domestic / Family
    "DOMESTIC VIOLENCE":                                "Domestic / Family",

    # Public Corruption
    "INTERFERENCE WITH PUBLIC OFFICER":                 "Public Corruption",

    # Other
    "NON-CRIMINAL":                                     "Other",
    "NON - CRIMINAL":                                   "Other",
    "NON-CRIMINAL (SUBJECT SPECIFIED)":                 "Other",
    "OTHER OFFENSE":                                    "Other",
    "RITUALISM":                                        "Other",
}