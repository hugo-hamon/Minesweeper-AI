import imageio.v2 as imageio
import time
import os


def png_to_gif(png_dir: str, gif_path: str) -> None:
    """Convert a directory of pngs to a gif"""
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_path, images, duration=0.5)


if __name__ == "__main__":
    png_dir = 'asset/screenshot'
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    gif_path = f'asset/gif/{int(time.time())}.gif'
    if not os.path.exists('asset/gif'):
        os.makedirs('asset/gif')

    png_to_gif(png_dir, gif_path)
    print(f"Saved gif to {gif_path}")
    for file_name in os.listdir(png_dir):
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            os.remove(file_path)
    print("Cleaned up pngs")
