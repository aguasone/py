import cv2

class Draw():
  def __init__(self):
    pass

  def face(self, image, r, d, startX, startY, endY, x1, x2, y1, y2, color, thickness, text, name):

    # Top left drawing
    cv2.line(image, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(image, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right drawing
    cv2.line(image, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(image, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left drawing
    cv2.line(image, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(image, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right drawing
    cv2.line(image, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(image, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.putText(image, text, (startX, startY),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

    cv2.putText(image, name, (x1 + r, endY),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)

    return image
