# Rule-based ISL gesture recognition
def classify_gesture(landmarks):
    # Example rules (dummy, you must train or define real ISL rules)
    finger_up = []

    # Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
    for tip in [4, 8, 12, 16, 20]:
        finger_up.append(landmarks[tip][1] < landmarks[tip - 2][1])  # tip above pip

    if finger_up == [False, True, False, False, False]:
        return "A"
    elif all(finger_up):
        return "B"
    elif finger_up == [False, True, True, False, False]:
        return "C"
    return None
