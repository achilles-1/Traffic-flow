def classify_traffic(traffic_volume):

    if traffic_volume < 3000:
        return "LOW"

    elif traffic_volume <= 6000:
        return "MEDIUM"

    else:
        return "HIGH"