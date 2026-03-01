from ultralytics import YOLO

# 1. Load your custom-trained weights
# Using 'r' before the string handles Windows file paths safely
model = YOLO(r"C:\Users\prasa\runs\detect\train3\weights\best.pt")

# 2. Pick your image (Grab a great photo of Virat Kohli and drop it in the folder!)
image_path = "test.jpg" 

# 3. Run the AI prediction
# save=True tells it to draw the boxes and save the image
# conf=0.5 means it will only show boxes if it's 50%+ confident
results = model.predict(source=image_path, save=True, conf=0.5)

# 4. Print out exactly what the AI saw in the terminal
print("\n--- AI ANALYSIS COMPLETE ---")
for result in results:
    print(f"I found {len(result.boxes)} object(s) in the image!")
    
    # Optional: Print the exact class names it found (like 'bat' or 'batsman')
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        print(f"- Detected: {class_name}")