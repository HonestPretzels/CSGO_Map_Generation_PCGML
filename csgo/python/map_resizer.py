from PIL import Image
import sys
import os

def resizeImage(scale, input, output):
    with Image.open(input) as image:
        resizedImage = image.resize((scale, scale))
        resizedImage.save(output)

def main():
    mapPath = sys.argv[1]
    outPath = sys.argv[2]
    maps = [f for f in os.listdir(mapPath) if os.path.isfile(os.path.join(mapPath,f))]
    for map in maps:
        resizeImage(128, os.path.join(mapPath, map), os.path.join(outPath, map))

if __name__ == "__main__":
    main()