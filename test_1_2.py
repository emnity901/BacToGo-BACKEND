import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.font_manager as fm

def text_to_image(latex_text, output_path="output.png"):
    """
    Convert LaTeX text to an image with precise height calculation.
    The image height will be exactly (number of lines × line height).
    """
    # Split text at first period
    if '.' in latex_text:
        parts = latex_text.split('.', 1)
        modified_text = parts[0] + '.\n' + parts[1].lstrip()
    else:
        modified_text = latex_text

    # Count the number of lines
    num_lines = modified_text.count('\n') + 1

    # Create a dummy figure to measure font metrics
    dummy_fig, dummy_ax = plt.subplots()
    dummy_text = dummy_ax.text(0, 0, 'Tg', fontsize=11)  # 'Tg' contains ascender and descender
    dummy_fig.canvas.draw()
    
    # Get font properties
    renderer = dummy_fig.canvas.get_renderer()
    bbox = dummy_text.get_window_extent(renderer)
    line_height = bbox.height / dummy_fig.dpi  # Line height in inches
    plt.close(dummy_fig)

    # Calculate exact figure height
    fig_height = num_lines * line_height * 1.05  # Small 5% buffer for safety

    # Create final figure with exact dimensions
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis('off')
    
    # Add text aligned to top-left
    text = ax.text(0, 1, modified_text, 
                  fontsize=11,
                  ha='left',
                  va='top',
                  wrap=False)

    # Adjust layout to remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save with tight bounding box
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    Image.open(buf).save(output_path)
    plt.close()
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    test_text = r"Se consideră funcția \( f : \mathbb{R} \rightarrow \mathbb{R}, \, f(x) = x^2 + bx + c \), unde \( b \) și \( c \) sunt numere reale. Determinați valorile reale ale lui \( b \) și \( c \), știind că graficul funcției \( f \) trece prin punctele \( A(1, 4) \) și \( B(-1, 6) \). Si dupa aceea blablabla blebleblebleblebleblelbelblelbelbleblelbelbelbe.lbelblelbelbleblelb"
    text_to_image(test_text)