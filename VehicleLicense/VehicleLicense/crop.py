from PIL import Image

for i in range(10, 99):

    img = Image.open(f"./Data/F/F_00{i}.jpg")
    print(img.size)
    cropped = img.crop((5, 0, 15, 20))  # (left, upper, right, lower)
    cropped.save(f"./Data/F/F_00{i}.jpg")