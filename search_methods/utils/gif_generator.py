import os
import re
from search_methods.src.node import Node
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from search_methods.src.sokoban import Sokoban, Symbol


def generate_gif(path, game: Sokoban, gif_name: str = "sokoban.gif"):
    print_path(path, game)
    create_gif(gif_name=gif_name)
    delete_frames()


def save_frame_with_matplotlib(sokoban: Sokoban, frame_number: int, output_folder: str = 'search_methods/frames'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    board = sokoban.level  # Asumimos que el atributo `level` contiene el tablero como una lista de listas
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, len(board[0]))
    ax.set_ylim(0, len(board))
    ax.set_aspect('equal')
    ax.set_axis_off()

    for y, line in enumerate(board):
        for x, char in enumerate(line):
            if char.value == Symbol.WALL.value:  # Muro
                ax.add_patch(patches.Rectangle((x, len(board) - y - 1), 1, 1, color='black'))
            elif char.value == Symbol.TARGET.value:  # Objetivo
                ax.add_patch(patches.Rectangle((x, len(board) - y - 1), 1, 1, color='lightgrey'))
            # elif char == Symbol.PLAYER_ON_TARGET:  # Jugador en objetivo
            #     ax.add_patch(patches.Circle((x + 0.5, len(board) - y - 1 + 0.5), 0.4, color='red'))
            elif char.value == Symbol.FREE.value:  # Espacio vacío
                ax.add_patch(patches.Rectangle((x, len(board) - y - 1), 1, 1, color='white'))
    for target in sokoban.targets:
        tx, ty = target
        ax.add_patch(patches.Rectangle((tx, len(board) - ty - 1), 1, 1, color='lightgrey'))

    px, py = sokoban.player_pos
    if sokoban.player_pos in sokoban.targets:
        ax.add_patch(patches.Circle((px + 0.5, len(board) - py - 1 + 0.5), 0.4, color='blue'))
    else:
        ax.add_patch(patches.Circle((px + 0.5, len(board) - py - 1 + 0.5), 0.4, color='red'))

    for box in sokoban.boxes:
        bx, by = box
        if box in sokoban.targets:
            ax.add_patch(patches.Rectangle((bx, len(board) - by - 1), 1, 1, color='brown'))
        else:
            ax.add_patch(patches.Rectangle((bx, len(board) - by - 1), 1, 1, color='green'))


    plt.savefig(f"{output_folder}/frame_{frame_number if frame_number >= 10 else f'0{frame_number}'}.png")
    plt.close(fig)  # Cerrar la figura para liberar memoria


def print_path(path: list[Node], sokoban: Sokoban):
    for i, node in enumerate(path):
        sokoban.boxes = node.state.boxes
        sokoban.player_pos = node.state.player_pos
        save_frame_with_matplotlib(sokoban, i)


def delete_frames(output_folder: str = 'search_methods/frames'):
    for img in os.listdir(output_folder):
        img_path = os.path.join(output_folder, img)
        if img.endswith(".png"):
            os.remove(img_path)

def natural_sort_key(s):
    # Esta función extrae los números del nombre del archivo y los convierte en enteros para una ordenación correcta
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def create_gif(output_folder: str = 'search_methods/frames', gif_name: str = 'sokoban_solution.gif'):
    frames = []
    imgs = sorted([img for img in os.listdir(output_folder) if img.endswith(".png")], key=natural_sort_key)

    for img in imgs:
        frame = Image.open(os.path.join(output_folder, img))
        frames.append(frame)

    if frames:
        # Guarda el GIF usando el primer frame como base y añadiendo los demás
        frames[0].save(f"search_methods/gifs/{gif_name}", format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=250,  # Duración entre frames en milisegundos
                       loop=0)  # 0 para loop infinito
        print(f"GIF saved as {gif_name}")
    else:
        print("No images found to generate the GIF.")


