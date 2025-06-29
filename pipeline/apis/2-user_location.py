#!/usr/bin/env python3
"""Script that prints the location of a GitHub user."""

import sys
import requests
import time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    url = sys.argv[1]
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(data.get("location"))
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = response.headers.get("X-RateLimit-Reset")
            if reset_time is not None:
                try:
                    reset_time = int(reset_time)
                    now = int(time.time())
                    minutes = int((reset_time - now) / 60)
                    if minutes < 0:
                        minutes = 0
                    print("Reset in {} min".format(minutes))
                except ValueError:
                    print("Reset in unknown time")
            else:
                print("Reset in unknown time")
        else:
            print("Error: HTTP {}".format(response.status_code))
    except requests.RequestException:
        print("Network error")

