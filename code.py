import fitz  # PyMuPDF
import json
import asyncio
import aiohttp
import ssl
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps
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
    # Only split if it IS problem 1.2
    if is_problem_1_2:
        if '.' in formatted_text:
            parts = formatted_text.split('.', 1)
            formatted_text = parts[0] + '.\n' + parts[1].lstrip()
    
    # The rest of the logic remains the same, but the split above now only happens for 1.2
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
    
    # Open and process the image using PIL to tightly crop the text
    image = Image.open(buf)
    
    # Define a helper function to trim whitespace using a threshold
    def trim(im, threshold=10):
        gray = im.convert("L")
        diff = ImageChops.difference(gray, Image.new("L", im.size, 255))
        diff = diff.point(lambda x: 0 if x < threshold else x)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return im

    image = trim(image)
    image.save(output_path)
    plt.close()
    return output_path

def replace_text_with_image(input_pdf, output_pdf, placeholder_text, image_path):
    doc = fitz.open(input_pdf)

    for page in doc:
        text_instances = page.search_for(placeholder_text)
        print(f"Page {page.number}: Searching for placeholder '{placeholder_text}' found {len(text_instances)} instances")

        for rect in text_instances:
            # Remove the original text
            page.add_redact_annot(rect, fill=(1, 1, 1))  # RGB white
            page.apply_redactions()

            # Open the image to get its dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()

            # Apply 0.3 scale to dimensions while maintaining aspect ratio
            scaled_width = img_width * 0.2
            scaled_height = img_height * 0.2

            # Calculate vertical center position
            placeholder_height = rect.height
            y_center = rect.y0 + (placeholder_height - scaled_height) / 2

            new_rect = fitz.Rect(
                rect.x0,                # Left-aligned X position
                y_center,               # Centered Y position
                rect.x0 + scaled_width, # Right edge from left
                y_center + scaled_height # Bottom edge from center
            )

            page.insert_image(new_rect, filename=image_path, overlay=True)

    doc.save(output_pdf)
    doc.close()

async def get_latex_text_from_api(session, subiect="1.1"):
    """Get LaTeX text from API"""
    with open("prompturi/master-prompt.txt", "r", encoding="utf-8") as file:
        master_prompt = file.read().strip()

    subiect_folder = f"prompturi/SUBIECTUL_{subiect}"
    subiect_files = os.listdir(subiect_folder)
    random_prompt_file = os.path.join(subiect_folder, random.choice(subiect_files))

    with open(random_prompt_file, "r", encoding="utf-8") as subiect_file:
        random_prompt = subiect_file.read().strip()

    prompt_text = f"{master_prompt}\n\n{random_prompt}"
    print(f"[Problem {subiect}] Using prompt from: {random_prompt_file}")
    print(f"[Problem {subiect}] Full prompt used:")
    print(prompt_text)

    async with session.post(
        url="https://api.novita.ai/v3/openai/chat/completions",
        headers={
            "Authorization": "Bearer sk_xS9HwAFlcHIZHGeHnoifr_Am2jovQx8U65bJVKLCF4Y",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen/qwen-2.5-72b-instruct",
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 512
        }
    ) as response:
        try:
            result = await response.json()
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

async def main():
    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Step 1: Get LaTeX text for all problems concurrently
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = [
            get_latex_text_from_api(session, "1.1"),
            get_latex_text_from_api(session, "1.2"),
            get_latex_text_from_api(session, "1.3"),
            get_latex_text_from_api(session, "1.4"),
            get_latex_text_from_api(session, "1.5")
        ]
        latex_texts = await asyncio.gather(*tasks)
    
    latex_text_1, latex_text_2, latex_text_3, latex_text_4, latex_text_5 = latex_texts
    
    # Step 2: Convert LaTeX texts to images
    image_path_1 = text_to_image(latex_text_1, "output1.png")
    image_path_2 = text_to_image(latex_text_2, "output2.png", is_problem_1_2=True)
    image_path_3 = text_to_image(latex_text_3, "output3.png")
    image_path_4 = text_to_image(latex_text_4, "output4.png", is_problem_1_2=True)
    image_path_5 = text_to_image(latex_text_5, "output5.png", is_problem_1_2=True)
    
    # Step 3: Replace placeholders in PDF with the images
    temp_pdf_1 = "temp_modified_1.pdf"
    temp_pdf_2 = "temp_modified_2.pdf"
    temp_pdf_3 = "temp_modified_3.pdf"
    
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
        output_pdf=temp_pdf_3,
        placeholder_text="{{problema1.3}}",
        image_path=image_path_3
    )
    
    replace_text_with_image(
        input_pdf=temp_pdf_3,
        output_pdf="modified_template.pdf",
        placeholder_text="{{problema1.4}}",
        image_path=image_path_4
    )
    
    temp_pdf_4 = "temp_modified_4.pdf"
    replace_text_with_image(
        input_pdf="modified_template.pdf",
        output_pdf=temp_pdf_4,
        placeholder_text="{{problema1.5}}",
        image_path=image_path_5
    )
    os.replace(temp_pdf_4, "final_template.pdf")
    final_pdf = "final_template.pdf"
    
    # Optional: Clean up the temporary files
    # os.remove(image_path_1)
    # os.remove(image_path_2)
    # os.remove(image_path_3)
    # os.remove(image_path_4)
    os.remove(temp_pdf_1)
    os.remove(temp_pdf_2)
    os.remove(temp_pdf_3)
    print("Process completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
