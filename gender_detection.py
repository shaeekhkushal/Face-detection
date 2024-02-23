import cv2
import face_recognition


def main():
	# Initialize the webcam
	cap = cv2.VideoCapture(0)

	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Find faces in the frame
		face_locations = face_recognition.face_locations(frame)

		# For each face found, classify gender
		for top, right, bottom, left in face_locations:
			face_image = frame[top:bottom, left:right]

			# Use face_recognition library to classify gender
			face_encoding = face_recognition.face_encodings(face_image)[0]
			results = face_recognition.face_gender([face_encoding])
			gender = "Male" if results[0] > 0.5 else "Female"

			# Draw bounding box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

			# Display gender label
			cv2.putText(frame, gender, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		# Display the resulting frame
		cv2.imshow('Face Detection with Gender Classification', frame)

		# Break the loop if 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the capture and close all OpenCV windows
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
