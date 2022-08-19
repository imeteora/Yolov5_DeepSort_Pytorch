# Area intrusion check
def checkAreaIntrusion(areas, objects):
    # global audio_enable_flag
    # global sound_thread_warning
    for area in areas:
        area.count = 0
        for obj in objects:
            if point_polygon_test(area.contour, obj.anchor_pt):
                area.count += 1


# Test whether the test_point is in the polygon or not - 指定の点がポリゴン内に含まれるかどうかを判定
# test_point = (x,y)
# polygon = collection of points  [ (x0,y0), (x1,y1), (x2,y2) ... ]
def point_polygon_test(polygon, test_point) -> bool:
    if len(polygon) < 3:
        return False
    prev_point = polygon[-1]  # Use the last point as the starting point to close the polygon
    line_count = 0
    for point in polygon:
        # Check if Y coordinate of the test point is in range
        if min(prev_point[1], point[1]) <= test_point[1] <= max(prev_point[1], point[1]):
            # delta_x / delta_y
            gradient = (point[0] - prev_point[0]) / (point[1] - prev_point[1])
            # Calculate X coordinate of a line
            line_x = prev_point[0] + (test_point[1] - prev_point[1]) * gradient
            if line_x < test_point[0]:
                line_count += 1
        prev_point = point
    # Check how many lines exist on the left to the test_point
    included = True if line_count % 2 == 1 else False
    return included
