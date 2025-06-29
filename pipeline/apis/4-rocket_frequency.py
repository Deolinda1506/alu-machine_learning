#!/usr/bin/env python3
"""Script to count the number of SpaceX launches per rocket."""

import requests


def get_rocket_launch_counts():
    """
    Count and display the number of launches for each SpaceX rocket.

    The output format is:
    <rocket name>: <number of launches>

    Sorted by launch count (descending), then rocket name (A-Z).
    """
    # Get all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(launches_url).json()

    # Count launches per rocket ID
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Get rocket ID-to-name mapping
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets = requests.get(rockets_url).json()
    id_to_name = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Build list of (name, count)
    rocket_list = []
    for rocket_id, count in rocket_counts.items():
        name = id_to_name.get(rocket_id, "Unknown")
        rocket_list.append((name, count))

    # Sort by count descending, then name ascending
    rocket_list.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in rocket_list:
        print("{}: {}".format(name, count))


if __name__ == "__main__":
    get_rocket_launch_counts()

