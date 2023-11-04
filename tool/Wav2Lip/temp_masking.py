pred = face_detector.get_landmarks(f[y1:y2, x1:x2])[0]

# print(pred)
pts = []
for i, (x,y) in enumerate(pred):
	if i > 26:
		break
	# print(x, y)
	pts.append([int(x), int(y)])
	# cv2.circle(f, (int(x), int(y)), 1, (255, 0, 0), 2)

pts = np.array(pts)
# print(pts)
pts[17:] = pts[17:][::-1]
rect = cv2.boundingRect(pts)
x, y, w, h = rect
# print(rect, p.shape)
cropped = np.copy(p)
# pts = pts - pts.min(axis=0)
mask = np.zeros(cropped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
dst = cv2.bitwise_and(cropped, cropped, mask=mask)
bg = np.ones_like(cropped, np.uint8)*255
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg + dst

frame_copy = np.copy(f).astype(np.float)
frame_copy[y1:y2, x1:x2][:, :, 0] = frame_copy[y1:y2, x1:x2][:, :, 0] - mask
frame_copy[y1:y2, x1:x2][:, :, 1] = frame_copy[y1:y2, x1:x2][:, :, 1] - mask
frame_copy[y1:y2, x1:x2][:, :, 2] = frame_copy[y1:y2, x1:x2][:, :, 2] - mask
frame_copy[frame_copy < 0] = 0.0

frame_copy[y1:y2, x1:x2] = frame_copy[y1:y2, x1:x2] + dst
cv2.imwrite('faces/mask_{:05d}.png'.format(idx), mask)
cv2.imwrite('faces/dst_{:05d}.png'.format(idx), dst)
cv2.imwrite('faces/dst2_{:05d}.png'.format(idx), dst2)
cv2.imwrite('faces/frame_copy_{:05d}.png'.format(idx), frame_copy)
cv2.imwrite('faces/cropped_{:05d}.png'.format(idx), cropped)
cv2.imwrite('faces/f_{:05d}.png'.format(idx), f)
idx += 1
# print(frame_copy)
# frame_copy = np.array(frame_copy, dtype=np.uint8)
fcc = cv2.medianBlur(frame_copy.astype(np.uint8),7)
f = fcc