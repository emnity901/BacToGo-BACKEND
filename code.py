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

def text_to_image(latex_text, output_path="output.png"):
    """Convert LaTeX text to an image with proper dimensions"""
    # Replace LaTeX inline delimiters with dollar signs for matplotlib
    formatted_text = latex_text.replace(r"\(", "$").replace(r"\)", "$")
    
    # Create a figure and an axis
    fig, ax = plt.subplots()
    
    # Render the text at the center
    text_obj = ax.text(0.5, 0.5, formatted_text, fontsize=11, ha='center', va='center')
    ax.axis('off')
    
    # Draw the canvas to get the text bounding box
    fig.canvas.draw()
    bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
    
    # Convert bounding box dimensions from pixels to inches
    bbox_inch = bbox.transformed(fig.dpi_scale_trans.inverted())
    
    # Define padding in inches
    pad = 0
    # Set figure size based on text bounding box plus padding
    fig.set_size_inches(bbox_inch.width + 2*pad, bbox_inch.height + 2*pad)
    
    # Save the image with tight bounding box
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=pad, dpi=300)
    buf.seek(0)
    
    # Open and save the image using PIL
    image = Image.open(buf)
    image.save(output_path)
    plt.close()  # Close the figure to free memory
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
                y_position,                  # Centered Y + offset
                x_left + img_width_scaled,  # Right edge
                y_position + img_height_scaled  # Bottom edge
            )
            
            page.insert_image(new_rect, filename=image_path)
    
    doc.save(output_pdf)
    doc.close()

def get_latex_text_from_api():
    """Get LaTeX text from API"""
    with open("prompturi/master-prompt.txt", "r") as file:
        master_prompt = file.read().strip()

    subiect_folder = "prompturi/SUBIECTUL_1.1"
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
    # Step 1: Get LaTeX text from API
    latex_text = get_latex_text_from_api()
    
    # Step 2: Convert LaTeX text to image
    image_path = text_to_image(latex_text)
    
    # Step 3: Replace placeholder in PDF with the image
    replace_text_with_image(
        input_pdf="template.pdf",
        output_pdf="modified_template.pdf",
        placeholder_text="{{problema1.1}}",
        image_path=image_path
    )
    
    # Optional: Clean up the temporary image file
    os.remove(image_path)
    print("Process completed successfully")
