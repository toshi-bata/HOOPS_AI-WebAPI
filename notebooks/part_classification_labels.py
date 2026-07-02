"""
Shared 45-class part label definitions for the FabWave Part Classification dataset.

This module is intentionally dependency-free so it can be imported by both:
- Jupyter notebooks (cad_tasks_custom_part_classification.py)
- The FastAPI application (core.py)

Canonical source: Do NOT duplicate this dict elsewhere.
"""

labels_description: dict[int, dict[str, str]] = {
    0:  {"name": "Bearings",                               "description": "FabWave dataset sample."},
    1:  {"name": "Bolts",                                  "description": "FabWave dataset sample."},
    2:  {"name": "Brackets",                               "description": "FabWave dataset sample."},
    3:  {"name": "Bushing",                                "description": "FabWave dataset sample."},
    4:  {"name": "Bushing_Damping_Liners",                 "description": "FabWave dataset sample."},
    5:  {"name": "Collets",                                "description": "FabWave dataset sample."},
    6:  {"name": "Gasket",                                 "description": "FabWave dataset sample."},
    7:  {"name": "Grommets",                               "description": "FabWave dataset sample."},
    8:  {"name": "HeadlessScrews",                         "description": "FabWave dataset sample."},
    9:  {"name": "Hex_Head_Screws",                        "description": "FabWave dataset sample."},
    10: {"name": "Keyway_Shaft",                           "description": "FabWave dataset sample."},
    11: {"name": "Machine_Key",                            "description": "FabWave dataset sample."},
    12: {"name": "Nuts",                                   "description": "FabWave dataset sample."},
    13: {"name": "O_Rings",                                "description": "FabWave dataset sample."},
    14: {"name": "Thumb_Screws",                           "description": "FabWave dataset sample."},
    15: {"name": "Pipe_Fittings",                          "description": "FabWave dataset sample."},
    16: {"name": "Pipe_Joints",                            "description": "FabWave dataset sample."},
    17: {"name": "Pipes",                                  "description": "FabWave dataset sample."},
    18: {"name": "Rollers",                                "description": "FabWave dataset sample."},
    19: {"name": "Rotary_Shaft",                           "description": "FabWave dataset sample."},
    20: {"name": "Shaft_Collar",                           "description": "FabWave dataset sample."},
    21: {"name": "Slotted_Flat_Head_Screws",               "description": "FabWave dataset sample."},
    22: {"name": "Socket_Head_Screws",                     "description": "FabWave dataset sample."},
    23: {"name": "Washers",                                "description": "FabWave dataset sample."},
    24: {"name": "Boxes",                                  "description": "FabWave dataset sample."},
    25: {"name": "Cotter_Pin",                             "description": "FabWave dataset sample."},
    26: {"name": "External Retaining Rings",               "description": "FabWave dataset sample."},
    27: {"name": "Eyesbolts With Shoulders",               "description": "FabWave dataset sample."},
    28: {"name": "Fixed Cap Flange",                       "description": "FabWave dataset sample."},
    29: {"name": "Gear Rod Stock",                         "description": "FabWave dataset sample."},
    30: {"name": "Gears",                                  "description": "FabWave dataset sample."},
    31: {"name": "Holebolts With Shoulders",               "description": "FabWave dataset sample."},
    32: {"name": "Idler Sprocket",                         "description": "FabWave dataset sample."},
    33: {"name": "Miter Gear Set Screw",                   "description": "FabWave dataset sample."},
    34: {"name": "Miter Gears",                            "description": "FabWave dataset sample."},
    35: {"name": "Rectangular Gear Rack",                  "description": "FabWave dataset sample."},
    36: {"name": "Routing EyeBolts Bent Closed Eye",       "description": "FabWave dataset sample."},
    37: {"name": "Sleeve Washers",                         "description": "FabWave dataset sample."},
    38: {"name": "Socket-Connect Flanges",                 "description": "FabWave dataset sample."},
    39: {"name": "Sprocket Taper-Lock Bushing",            "description": "FabWave dataset sample."},
    40: {"name": "Strut Channel Floor Mount",              "description": "FabWave dataset sample."},
    41: {"name": "Strut Channel Side-Side",                "description": "FabWave dataset sample."},
    42: {"name": "Tag Holder",                             "description": "FabWave dataset sample."},
    43: {"name": "Webbing Guide",                          "description": "FabWave dataset sample."},
    44: {"name": "Wide Grip External Retaining Ring",      "description": "FabWave dataset sample."},
}
