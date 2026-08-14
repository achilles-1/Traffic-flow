def optimize_traffic(traffic):

    if traffic < 3000:
        level = "Low"
        green_time = 30
        red_time = 60

    elif traffic < 6000:
        level = "Medium"
        green_time = 60
        red_time = 45

    else:
        level = "High"
        green_time = 90
        red_time = 30

    return level, green_time, red_time


if __name__ == "__main__":

    traffic = float(
        input("Enter predicted traffic volume: ")
    )

    level, green_time, red_time = optimize_traffic(
        traffic
    )

    print()
    print("===== Traffic Flow Optimization =====")
    print("Predicted Traffic:", round(traffic, 2))
    print("Traffic Level:", level)
    print("Recommended Green Time:", green_time, "seconds")
    print("Recommended Red Time:", red_time, "seconds")