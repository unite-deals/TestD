import streamlit as st
import numpy as np
from PIL import Image, ImageDraw


input_points = []
IMG_SIZE = 512
input_image = None

def generate_app(get_processed_inputs, inpaint):
    global input_points
    global input_image

    # Functions for handling clicks and running SAM
    def get_points(x, y, img):
        global input_image
        if len(input_points) == 0:
            input_image = img.copy()

        input_points.append([x, y])
        
        sam_output = run_sam()
        # Mark selected points with a green crossmark
        draw = ImageDraw.Draw(img)
        size = 10
        for point in input_points:
            x, y = point
            draw.line((x - size, y, x + size, y), fill="green", width=5)
            draw.line((x, y - size, x, y + size), fill="green", width=5)

        return sam_output, img

    def run_sam():
        if input_image is None:
            st.error("No points provided. Click on the image to select the object to segment with SAM")
            return None, None

        try:
            mask = get_processed_inputs(input_image, [input_points])
            res_mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))
            return (
                input_image.resize((IMG_SIZE, IMG_SIZE)), 
                [
                    (res_mask, "background"), 
                    (~res_mask, "subject")
                ]
            )
        except Exception as e:
            st.error(str(e))
            return None, None

    def run_inpaint(prompt, negative_prompt, cfg, seed, invert):
        if input_image is None:
            st.error("No points provided. Click on the image to select the object to segment with SAM")
            return None

        amask = run_sam()[1][0][0]

        if bool(invert):
            what = 'subject'
            amask = ~amask
        else:
            what = 'background'

        st.info(f"Inpainting {what}... (this will take up to a few minutes)")
        try:
            inpainted = inpaint(input_image, amask, prompt, negative_prompt, seed, cfg)
            return inpainted.resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            st.error(str(e))
            return None

    def reset_points():
        input_points.clear()

    def preprocess(input_img):
        if input_img is None:
            return None

        width, height = input_img.size
        if width != height:
            st.warning("Image is not square, adding white padding")
            new_size = max(width, height)
            new_image = Image.new("RGB", (new_size, new_size), 'white')
            left = (new_size - width) // 2
            top = (new_size - height) // 2
            new_image.paste(input_img, (left, top))
            input_img = new_image

        return input_img.resize((IMG_SIZE, IMG_SIZE))

    # Streamlit UI
    st.title("Image Inpainting")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        input_image = Image.open(uploaded_image)
        input_image = preprocess(input_image)

        # Display image for interaction
        st.image(input_image, caption="Input Image", use_column_width=True)

        # Select points (use Streamlit click functionality if needed)
        if st.button("Select Points"):
            # Sample: You can integrate mouse clicks to choose points here
            st.write("Click points on the image to segment.")

    # Inputs for inpainting
    prompt = st.text_input("Prompt for infill")
    negative_prompt = st.text_input("Negative prompt")
    cfg = st.slider("Classifier-Free Guidance Scale", 0.0, 20.0, 7.0, 0.05)
    random_seed = st.number_input("Random seed", value=74294536)
    invert_mask = st.checkbox("Infill subject instead of background")

    # Reset points button
    if st.button("Reset Points"):
        reset_points()

    # Run SAM and Inpainting
    if st.button("Run Inpaint"):
        if uploaded_image:
            result = run_inpaint(prompt, negative_prompt, cfg, random_seed, invert_mask)
            if result:
                st.image(result, caption="Inpainted Image", use_column_width=True)

# Example function call
generate_app(get_processed_inputs=lambda img, points: np.ones((IMG_SIZE, IMG_SIZE), dtype=bool), inpaint=lambda img, mask, prompt, neg_prompt, seed, cfg: img)
