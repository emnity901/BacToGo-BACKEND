import fitz  # PyMuPDF
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def text_to_image(latex_text, output_path="output.png", is_problem_1_2=False):
    """Convert LaTeX text to an image with proper dimensions"""
    # Replace LaTeX inline delimiters with dollar signs for matplotlib
    formatted_text = latex_text.replace(r"\(", "$").replace(r"\)", "$")
    if '.' in formatted_text:
        parts = formatted_text.split('.', 1)
        formatted_text = parts[0] + '.\n' + parts[1].lstrip()
    
    if is_problem_1_2:
        # Special handling for problem 1.2 with precise height calculation
        num_lines = formatted_text.count('\n') + 1

        # Create a dummy figure to measure font metrics
        dummy_fig, dummy_ax = plt.subplots()
        dummy_text = dummy_ax.text(0, 0, 'Tg', fontsize=11)
        dummy_fig.canvas.draw()

        # Get font properties
        renderer = dummy_fig.canvas.get_renderer()
        bbox = dummy_text.get_window_extent(renderer)
        line_height = bbox.height / dummy_fig.dpi  # Line height in inches
        plt.close(dummy_fig)

        # Calculate exact figure height
        fig_height = num_lines * line_height * 1.05  # Small 5% buffer

        # Create final figure with exact dimensions
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.axis('off')

        # Add text aligned to top-left
        text_obj = ax.text(0, 1, formatted_text,
                           fontsize=11,
                           ha='left',
                           va='top',
                           wrap=False)

        # Adjust layout to remove all margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        # Original handling for other problems
        fig, ax = plt.subplots()
        text_obj = ax.text(0.5, 0.5, formatted_text, fontsize=11, ha='center', va='center')
        ax.axis('off')

        # Draw the canvas to get the text bounding box
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_inch = bbox.transformed(fig.dpi_scale_trans.inverted())

        # Set figure size based on text bounding box
        fig.set_size_inches(bbox_inch.width, bbox_inch.height)
    
    # Save the image with tight bounding box
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)
    
    # Open and save the image using PIL
    image = Image.open(buf)
    image.save(output_path)
    plt.close()
    return output_path

def replace_text_with_image(input_pdf, output_pdf, placeholder_text, image_path):
    doc = fitz.open(input_pdf)

    for page in doc:
        text_instances = page.search_for(placeholder_text)

        for rect in text_instances:
            # Remove the original text
            page.add_redact_annot(rect, fill=(1, 1, 1))  # RGB white
            page.apply_redactions()

            # Open the image to get its dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()

            # Target height: 3x the approximate height of 12pt text
            base_height = 12 * (72 / 96)  # Base height for 12pt text
            target_height = base_height * 1  # 3x larger

            # Calculate scaling factor to match target height
            scale = target_height / img_height

            # Apply scaling to both dimensions (preserve aspect ratio)
            img_width_scaled = img_width * scale
            img_height_scaled = img_height * scale

            # Position the image at the LEFT edge of the placeholder
            x_left = rect.x0  # Start from left edge

            # Vertical position: Centered + OFFSET (adjust this value as needed)
            y_offset = 0  # Move down by 5 points (increase for more spacing)
            y_position = rect.y0 + (rect.height - img_height_scaled) / 2 + y_offset

            new_rect = fitz.Rect(
                x_left,                     # Left-aligned X
                y_position,                 # Centered Y + offset
                x_left + img_width_scaled,  # Right edge
                y_position + img_height_scaled  # Bottom edge
            )

            page.insert_image(new_rect, filename=image_path)

    doc.save(output_pdf)
    doc.close()

def get_latex_text_from_api(subiect="1.1"):
    """Get LaTeX text from API"""
    with open("prompturi/master-prompt.txt", "r") as file:
        master_prompt = file.read().strip()

    subiect_folder = f"prompturi/SUBIECTUL_{subiect}"
    subiect_files = os.listdir(subiect_folder)
    random_prompt_file = os.path.join(subiect_folder, random.choice(subiect_files))

    with open(random_prompt_file, "r") as subiect_file:
        random_prompt = subiect_file.read().strip()

    prompt_text = f"{master_prompt}\n\n{random_prompt}"

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
        })
    )

    try:
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            replacement_text = result["choices"][0]["message"]["content"]
            print("\nSuccessfully received API response")
            print("Extracted Content:", replacement_text)
            return replacement_text
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            return "Error: Unable to get proper response from API"
    except Exception as e:
        print(f"Error parsing response: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Step 1: Get LaTeX text for all three problems
    latex_text_1 = get_latex_text_from_api("1.1")
    latex_text_2 = get_latex_text_from_api("1.2")
    latex_text_3 = get_latex_text_from_api("1.3")
    
    # Step 2: Convert LaTeX texts to images
    image_path_1 = text_to_image(latex_text_1, "output1.png")
    image_path_2 = text_to_image(latex_text_2, "output2.png", is_problem_1_2=True)
    image_path_3 = text_to_image(latex_text_3, "output3.png")
    
    # Step 3: Replace placeholders in PDF with the images
    temp_pdf_1 = "temp_modified_1.pdf"
    temp_pdf_2 = "temp_modified_2.pdf"
    
    replace_text_with_image(
        input_pdf="template.pdf",
        output_pdf=temp_pdf_1,
        placeholder_text="{{problema1.1}}",
        image_path=image_path_1
    )
    
    replace_text_with_image(
        input_pdf=temp_pdf_1,
        output_pdf=temp_pdf_2,
        placeholder_text="{{problema1.2}}",
        image_path=image_path_2
    )
    
    replace_text_with_image(
        input_pdf=temp_pdf_2,
        output_pdf="modified_template.pdf",
        placeholder_text="{{problema1.3}}",
        image_path=image_path_3
    )
    
    # Optional: Clean up the temporary files
    os.remove(image_path_1)
    os.remove(image_path_2)
    os.remove(image_path_3)
    os.remove(temp_pdf_1)
    os.remove(temp_pdf_2)
    print("Process completed successfully")
