"""
Default 25-class MFR (Manufacturing Feature Recognition) label definitions.

Used as the fallback when HOOPS_AI_MFR_LABELS_DESCRIPTION / labels_description
is not set in the environment.  Override with the env variable when using a
custom MFR dataset with different classes.
"""

labels_description: dict[int, dict[str, str]] = {
    0:  {"name": "no-label",                              "description": "No label assigned."},
    1:  {"name": "rectangular_through_slot",              "description": "This is a rectangular MFR feature."},
    2:  {"name": "triangular_through_slot",               "description": "Triangular through-slot feature."},
    3:  {"name": "rectangular_passage",                   "description": "Rectangular passage feature."},
    4:  {"name": "triangular_passage",                    "description": "Triangular passage feature."},
    5:  {"name": "6sides_passage",                        "description": "Six-sided passage feature."},
    6:  {"name": "rectangular_through_step",              "description": "Rectangular through-step feature."},
    7:  {"name": "2sides_through_step",                   "description": "Two-sided through-step feature."},
    8:  {"name": "slanted_through_step",                  "description": "Slanted through-step feature."},
    9:  {"name": "rectangular_blind_step",                "description": "Rectangular blind-step feature."},
    10: {"name": "triangular_blind_step",                 "description": "Triangular blind-step feature."},
    11: {"name": "rectangular_blind_slot",                "description": "Rectangular blind-slot feature."},
    12: {"name": "rectangular_pocket",                    "description": "Rectangular pocket feature."},
    13: {"name": "triangular_pocket",                     "description": "Triangular pocket feature."},
    14: {"name": "6sides_pocket",                         "description": "Six-sided pocket feature."},
    15: {"name": "chamfer",                               "description": "Chamfer feature."},
    16: {"name": "circular through slot",                 "description": "Circular through-slot feature."},
    17: {"name": "through hole",                          "description": "Description for through hole."},
    18: {"name": "circular blind step",                   "description": "Description for circular blind step."},
    19: {"name": "horizontal circular end blind slot",    "description": "Description for horizontal circular end blind slot."},
    20: {"name": "vertical circular end blind slot",      "description": "Description for vertical circular end blind slot."},
    21: {"name": "circular end pocket",                   "description": "Description for circular end pocket."},
    22: {"name": "o-ring",                                "description": "Description for o-ring."},
    23: {"name": "blind hole",                            "description": "Description for blind hole."},
    24: {"name": "fillet",                                "description": "Description for fillet."},
}
