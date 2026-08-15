def optimize_traffic(traffic_level):

    if traffic_level == "Low":
        green_time = 30
        red_time = 60

    elif traffic_level == "Medium":
        green_time = 60
        red_time = 45

    elif traffic_level == "High":
        green_time = 90
        red_time = 30

    else:
        raise ValueError("Invalid traffic level")

    return green_time, red_time


if __name__ == "__main__":

    traffic_level = input(
        "Enter traffic level (Low/Medium/High): "
    )

    green_time, red_time = optimize_traffic(
        traffic_level
    )

    print()
    print("===== Traffic Flow Optimization =====")
    print("Traffic Level:", traffic_level)
    print("Recommended Green Time:", green_time, "seconds")
    print("Recommended Red Time:", red_time, "seconds")