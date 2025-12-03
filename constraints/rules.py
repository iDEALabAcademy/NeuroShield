

"""
Symbolic Knowledge Base for GTSRB Traffic Signs

This module defines the ground truth symbolic rules about traffic sign attributes.
These rules encode human knowledge about traffic sign structure that should be 
universally true (e.g., "all stop signs are octagons").

The neural network is trained to respect these constraints through soft logic losses.
"""

# === SHAPE MAP ===
# Maps each traffic sign class to its geometric shape
# Example: class 14 (stop sign) -> 'octagon'
class_shape_map = {
    **{i: 'circle' for i in [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]},
    **{i: 'triangle' for i in [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},
    12: 'diamond',
    14: 'octagon'
}

shape_to_id = {'circle': 0, 'triangle': 1, 'octagon': 2, 'diamond': 3, 'other': 4}

# === COLOR PARTS ===
# Maps each class to its color components: border, fill (background), and item (icon/text)
# Example: class 14 (stop) -> {'border': 'white', 'fill': 'red', 'item': 'white'}
# This captures that stop signs have red fill, white border, and white text
class_color_parts = {
    # Speed limits
    **{i: {'border': 'red', 'fill': 'white', 'item': 'black'} for i in [0,1,2,3,4,5,7,8]},
    6: {'border': 'black', 'fill': 'white', 'item': 'black'},
    
    # Prohibitions
    9: {'border': 'red', 'fill': 'white', 'item': 'black'},
    10: {'border': 'red', 'fill': 'white', 'item': 'black'},
    15: {'border': 'red', 'fill': 'white', 'item': 'none'},
    16: {'border': 'red', 'fill': 'white', 'item': 'black'},
    17: {'border': 'red', 'fill': 'white', 'item': 'none'},
    32: {'border': 'black', 'fill': 'white', 'item': 'none'},
    41: {'border': 'black', 'fill': 'white', 'item': 'black'},
    42: {'border': 'black', 'fill': 'white', 'item': 'black'},

    # Priority
    11: {'border': 'red', 'fill': 'white', 'item': 'black'},
    12: {'border': 'white', 'fill': 'yellow', 'item' : 'none'},
    13: {'border': 'red', 'fill': 'white', 'item': 'none'},
    14: {'border': 'white', 'fill': 'red', 'item': 'white'},

    # Warnings (triangle)
    **{i: {'border': 'red', 'fill': 'white', 'item': 'black'} for i in [18,21,22,23,24,25,26,27,28,29,30,31]},
    19: {'border': 'red', 'fill': 'white', 'item': 'black'},  # Curve Left
    20: {'border': 'red', 'fill': 'white', 'item': 'black'},

    # Mandatory
    **{i: {'border': 'blue', 'fill': 'blue', 'item': 'white'} for i in [33,34,35,36,37,38,39,40]},
}

color_label_to_id = {
    'red': 0,
    'white': 1,
    'blue': 2,
    'yellow': 3,
    'black': 4,
    'none': 5,
    'other': 6,  # Fallback for undefined colors
}

# === CATEGORY ===
# Groups traffic signs into functional categories
# Example: classes 0-8 are all speed limit signs
class_category_map = {
    **{i: 'speed' for i in range(0, 9)},
    **{i: 'prohibition' for i in [9,10,15,16,17,32,41,42]},
    **{i: 'warning' for i in range(18, 32)},
    **{i: 'mandatory' for i in range(33, 41)},
    11: 'priority',
    12: 'priority',
    13: 'priority',
    14: 'priority',
}

category_to_id = {
    'speed': 0,
    'prohibition': 1,
    'mandatory': 2,
    'warning': 3,
    'information': 4,
    'priority': 5,
    'other': 6
}

# === ICON TYPE ===
# Describes what kind of icon/symbol appears on the sign
# Example: class 27 (pedestrians) -> 'human'
# Example: class 0-8 (speed limits) -> 'number'
class_icon_type_map = {
    # Vehicles
    9: 'vehicle', 10: 'vehicle', 16: 'vehicle', 41: 'vehicle', 42: 'vehicle',
    23: 'vehicle',

    #Numbers
    0: 'number', 1: 'number', 2: 'number', 3: 'number', 4: 'number',
    5: 'number', 6: 'number', 7: 'number', 8: 'number',


    # Humans
    25: 'human', 27: 'human', 28: 'human',

    # Arrows
    11: 'arrow',
    **{i: 'arrow' for i in range(33, 41)},  # mandatory

    # Other
    29: 'bicycle',
    30: 'snowflake',
    31: 'animal',
    18: 'text', 14: 'text',  # general caution with text
    26: 'misc',
    22: 'misc',
    19: 'road', 20: 'road', 21: 'road', 24: 'road',

    # No icon
    **{i: 'none' for i in [12,13,15,17,32]}
}

icon_type_to_id = {
    'human': 0,
    'vehicle': 1,
    'arrow': 2,
    'bicycle': 3,
    'animal': 4,
    'snowflake': 5,
    'road': 6,
    'text': 7,
    'misc': 8,
    'bump': 9,
    'number':10,
    'none': 11
}
