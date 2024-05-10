#finallllllllllllllllll


import cv2
import numpy as np
import gradio as gr

def detect_circles_and_rows_columns(image1, image2):
    # Function to detect circles
    def detect_circles(image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Hough Transform for circle detection
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=40, maxRadius=67)

        # Ensure at least some circles were found
        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # Draw the circles and print radius
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Green circles
                cv2.putText(image, f'Radius: {r}', (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Print radius
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Red center

            # Print the number of detected circles
            num_circles = len(circles)
            cv2.putText(image, f'Number of circles: {num_circles}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            print("No circles detected.")

        return num_circles, image

    # Function to detect rows and columns
    def detect_rows_and_columns(image):
        # Resize the image to fit within the display window
        max_height = 800  # Set maximum height
        max_width = 1200  # Set maximum width

        # Get the original height and width of the image
        height, width = image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Resize the image while maintaining the aspect ratio
        if width > max_width or height > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            image = cv2.resize(image, (new_width, new_height))

        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Hough line transform to detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        # Filter out horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) > abs(x2 - x1):
                    horizontal_lines.append(line)
                else:
                    vertical_lines.append(line)

        # Draw all detected lines
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        verline=len(vertical_lines)/2

        # Return the number of detected rows and columns
        num_rows = len(horizontal_lines)
        num_columns = len(vertical_lines)

        return num_rows, num_columns, image

    # Call circle detection function on first image
    num_circles, result_image1 = detect_circles(image1)

    # Call rows/columns detection function on second image
    num_rows, num_columns, result_image2 = detect_rows_and_columns(image2)

    # Return results in Gradio format
    return f'Number of circles: {num_circles}',result_image1, f'Detected Rows: {num_rows-1}, Total cylinders : {(num_rows-1)*num_circles}',  result_image2

# Interface
#inputs = [
#    gr.inputs.Image(label="Input Image 1"),
#    gr.inputs.Image(label="Input Image 2")
#]
#outputs = [
#    gr.outputs.Image(label="Output Image 1"),
#    gr.outputs.Textbox(label="Circle Detection Results"),
#    gr.outputs.Image(label="Output Image 2"),
#    gr.outputs.Textbox(label="Row and Column Detection Results")
#]

#gr.Interface(detect_circles_and_rows_columns, inputs, outputs, title="Circle and Row/Column Detector",
            # description="Upload two images to detect circles in the first image and rows/columns in the second image.").launch()

gr.Interface(fn=detect_circles_and_rows_columns, inputs=["image", "image"], outputs=["text", "image", "text", "image"], title="Gas Cylinders and Lines Detection").launch()
