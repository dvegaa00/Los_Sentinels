import os

def get_part_zones(root, part):
    """Return a list of (year, zona_x) that contain the given part (part_1 or part_2)"""
    result = []

    for year in sorted(os.listdir(root)):
        year_path = os.path.join(root, year)
        if not os.path.isdir(year_path): continue
        for zona in os.listdir(year_path):
            if not zona.startswith("zona"): continue
            part_path = os.path.join(year_path, zona, part)
            if os.path.exists(part_path):
                result.append((os.path.join(year_path, zona, part), part))
    return result