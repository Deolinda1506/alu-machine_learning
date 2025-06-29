#!/usr/bin/env python3
"""Script to print the next upcoming SpaceX launch in a specific format.

Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests
import datetime


def get_upcoming_launch():
    """
    Fetch and display the next SpaceX launch information.

    It shows the launch name, local date, rocket name, launchpad name,
    and launchpad locality in the required format.
    """
    try:
        # Fetch upcoming launches
        launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
        launches = requests.get(launches_url).json()

        # Sort by soonest launch time (UNIX timestamp)
        launches.sort(key=lambda x: x.get("date_unix", float("inf")))

        # Select the first (soonest) launch
        launch = launches[0]

        # Extract basic launch data
        launch_name = launch.get("name", "Unknown Launch")
        date_utc = launch.get("date_utc")
        rocket_id = launch.get("rocket")
        launchpad_id = launch.get("launchpad")

        # Convert UTC date to local ISO 8601 format
        date_obj = datetime.datetime.strptime(
            date_utc, "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        local_date = date_obj.astimezone().isoformat()

        # Fetch rocket name
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
        rocket = requests.get(rocket_url).json()
        rocket_name = rocket.get("name", "Unknown Rocket")

        # Fetch launchpad details
        launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(
            launchpad_id
        )
        launchpad = requests.get(launchpad_url).json()
        launchpad_name = launchpad.get("name", "Unknown Launchpad")
        locality = launchpad.get("locality", "Unknown Locality")

        # Final output
        output = "{} ({}) {} - {} ({})".format(
            launch_name, local_date, rocket_name, launchpad_name, locality
        )
        print(output)

    except Exception:
        print("An error occurred while fetching launch data.")


if __name__ == "__main__":
    get_upcoming_launch()
