# imports
import streamlit as st
import base64
import io
from io import BytesIO
import requests
import urllib
import numpy as np
import traceback

from clarifai.client.model import Model
from clarifai.modules.css import ClarifaiStreamlitCSS
from annotated_text import annotated_text
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_select import image_select

# streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# PAT = st.secrets["CLARIFAI_PAT"]
PAT = "176ef51de04a41c8a83f6e3c8ed64dfe"

##########################
#### HELPER FUNCTIONS ####
##########################

def text_color_for_background(hex_code):
    """Determine the appropriate text color (white or black) for a given background color."""
    return "#000000" if is_light_or_dark(hex_code) == "light" else "#ffffff"

def footer(st):
    with open('footer.html', 'r') as file:
        footer = file.read()
        st.write(footer, unsafe_allow_html=True)

def url_picture_to_base64(img_url):
    response = requests.get(img_url)
    return response.content

def display_segmented_image(pred_response, SEGMENT_IMAGE_URL):
    """Displays the segmented part of the image using the model response."""
    try:
        # Load original image using PIL
        response = requests.get(SEGMENT_IMAGE_URL)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Ensure image is in RGB format
        img_array = np.array(img)

        # Extract regions
        regions = pred_response.outputs[0].data.regions
        masks = []
        concepts = []

        # Check if any regions were detected
        if not regions:
            st.text("No cracks found")
            return

        for region in regions:
            masks.append(np.array(Image.open(BytesIO(region.region_info.mask.image.base64))))
            concepts.append(region.data.concepts[0])

        # Create overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Combine all masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask > 0)

        # Convert combined mask to image
        mask_image = Image.fromarray((combined_mask * 255).astype('uint8'))
        
        # Create green overlay
        green_overlay = Image.new('RGBA', img.size, (0, 255, 0, 102))
        
        # Apply mask to green overlay
        final_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        final_overlay.paste(green_overlay, mask=mask_image)

        # Composite the images
        result = Image.alpha_composite(img.convert('RGBA'), final_overlay)
        result = result.convert('RGB')

        # Display the result using Streamlit
        st.image(result, use_column_width=True)

        # Display confidence scores using annotated_text
        annotation_data = []
        tag_bg_color_1 = "#00815f"
        tag_text_color_1 = "#ffffff"

        for concept in concepts:
          percentage = concept.value * 100
          annotation_data.append(
                (f'{concept.name}', f'{percentage:.2f}%', tag_bg_color_1, tag_text_color_1))

        list_with_empty_strings = []
        for item in annotation_data:
            list_with_empty_strings.append(item)
            list_with_empty_strings.append(" ")

        if list_with_empty_strings and list_with_empty_strings[-1] == " ":
            list_with_empty_strings.pop()

        st.write("Detected Defect Region:")
        annotated_text(*tuple(list_with_empty_strings))

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

####################
####  SIDEBAR   ####
####################

with st.sidebar:
    st.caption('Below options are mostly here to help customize any displayed graphics/text.')

    with st.expander('Header Setup'):
        company_logo = st.text_input(label='Banner Url', value='https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png')
        company_logo_width = st.slider(label='Banner Width', min_value=1, max_value=1000, value=300)
        page_title = st.text_input(label='Module Title', value='Visual Inspection Demo')

    with st.expander('Anamoly Detection'):
        anamoly_detection_subheader_title = st.text_input(label='Anamoly Detection subheader title', value='✨ Leveraging Clarifai for Anamoly Detection ✨')
        anamoly_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/d364a803a427d30363defabb5ac9cdb6?t=1683122597793\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/aafd51c3511a17aad284987942cdb5c2?t=1683122612382\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/bce40a06cb39ecd513d277ecdf3ab0a7?t=1683128606787\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/f3f74ebfec4d67b3959cde25575161a0?t=1683122628342'
        )

    with st.expander('Insulator Defect Detection'):
        defect_detection_subheader_title = st.text_input(label='Insulator Defect Detection Subheader Text', value='✨ Leveraging Clarifai for Insulator Defect Detection ✨')
        defect_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_1.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_2.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_3.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_4.jpeg'
        )
        box_color = st.color_picker(label='Detection Bounding box Color', value='#0000FF', key='color')
        box_thickness = st.slider(label='Detection Bounding box Thickness', min_value=1, max_value=10, value=3)

        insulator_defect_threshold = st.slider(label='Insulator Defect Threshold', min_value=0.0, max_value=1.0, value=0.3)
        tag_bg_color_1 = st.color_picker(label='Tag Background Color', value='#aabbcc', key='tag_bg_color_1')
        tag_text_color_1 = st.color_picker(label='Tag Text Color', value='#2B2D37', key='tag_text_color_1')

    with st.expander('Crack Segmentation'):
        segmentation_subheader_title = st.text_input(label='Crack Segmentation Subheader Text', value='✨ Leveraging Clarifai to segment cracked parts ✨')
        crack_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_1.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_2.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_3.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_4.jpeg'
        )

    with st.expander('Surface Defect Classification'):
        surface_defect_detection_subheader_title = st.text_input(label='Surface Defect Classification Subheader Text', value='✨ Leveraging Clarifai for Metal Surface Defect Classification ✨')
        surface_images = st.text_area(
            height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_2.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_3.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_4.png',
          )
        surface_defect_threshold = st.slider(label='Surface Defect Threshold', min_value=0.0, max_value=1.0, value=0.5)


        st.subheader('Output Display Options')
        tag_bg_color_2 = st.color_picker(label='Tag Background Color', value='#aabbcc', key='tag_bg_color_2')
        tag_text_color_2 = st.color_picker(label='Tag Text Color', value='#2B2D37', key='tag_text_color_2')

####################
####  MAIN PAGE ####
####################

st.image(company_logo, width=company_logo_width)
st.title(page_title)

tab1, tab2, tab3, tab4 = st.tabs(['Anamoly Detection', 'Insulator Defect Detection', 'Crack Segmentation', 'Surface Defect Classification'])

##############################
#### Anamoly Detection ####
##############################

with tab1:
    try:
        st.subheader(anamoly_detection_subheader_title)
        
        # Simple, clear descriptions for each image
        image_descriptions = [
            "Chipped Tablet #1 - Minor chip on the tablet’s edge.",
            "Chipped Tablet #2 - Noticeable chip with a larger piece missing.",
            "Chipped Tablet #3 - Major break with multiple pieces missing.",
            "Dirty Surface - Visible dirt or residue on the tablet."
        ]
        
        img = image_select(
            label="Select an image to check:",
            images=anamoly_images.split('\n'),
            captions=image_descriptions
        )

        if st.button("Run Anomaly Detection"):
            st.divider()
            model_url = "https://clarifai.com/clarifai/anomaly-detection-tablet-pills/models/pill-anomaly"
            
            with st.spinner("Processing anomaly detection..."):
                model = Model(url=model_url, pat=PAT)
                res_pmo = model.predict_by_url(img, input_type="image")
                print(res_pmo)
                output_heatmap = res_pmo.outputs[0].data.heatmaps[0].base64
                heatmap_im = Image.open(BytesIO(output_heatmap))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write('Original Image')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)
                    st.caption("""
                    The raw image of the tablet as captured.
                    """)
                
                with col2:
                    st.write('Heat Map')
                    heatmap_im_color = ImageOps.colorize(heatmap_im, black='red', white='black')
                    heatmap_im_color = heatmap_im_color.resize(im1_pil.size, resample=0)
                    st.image(heatmap_im_color)
                    st.caption("""
                    Visual representation of detected anomalies:
                    - Darker red areas indicate higher likelihood of defects
                    - Lighter/Black areas represent normal/expected conditions
                    """)
                
                with col3:
                    # st.write("Combined View")
                    st.write("Composite")
                    mask = Image.new("L", im1_pil.size, 64)
                    composite_im = Image.composite(im1_pil, heatmap_im_color, mask)
                    st.image(composite_im)
                    st.caption("""
                    Overlay of original image and heatmap:
                    Shows both the tablet and the problem areas together.
                    Helps you see exactly where the defects are.
                    """)

        # Collapsible text field at the bottom
        with st.expander("Details"):
            st.markdown("""
            ### About Anomaly Detection
            Anomaly detection automatically identifies unusual patterns or defects in tablet pills that deviate from expected normal conditions. This helps in quality control by detecting manufacturing defects like chips, cracks, or contamination.
            
            ### How This Works
            1. **Image Selection**: 
                - Choose one of the tablet pill images from the Carousel
                - Each image shows different types of defects (chips, breaks, or dirt)
            
            2. **Output**:
                - Click "Run Anomaly Detection" to analyze the selected image
                - You'll see three views:
                    * Original Image: The unprocessed tablet photo
                    * Heat Map: Shows detected anomalies where:
                        - Darker red areas = Higher likelihood of defects
                        - Lighter/Black areas = Normal conditions
                    * Composite: Combines original image with heatmap for easy defect location
            
            ### Original Model
            This implementation uses Clarifai's Anomaly Detection model for tablet pills. 
            - View the App here: [Clarifai Pill Anomaly Detection](https://clarifai.com/clarifai/anomaly-detection-tablet-pills)
            """)
            
            # Original text area for additional notes
            project_details = st.text_area(
                "Add Your Notes:",
                height=100,
                placeholder="Add any additional notes..."
            )
            if project_details:
                st.markdown("### Your Notes:")
                st.write(project_details)
                            
    except Exception as e:
    #   st.error(f"Error in Anomaly Detection tab: {str(e)}")
        st.error(f"Error in Defect Detection tab: {str(e)}\n\n{traceback.format_exc()}")


#########################
#### Defect Detection ####
#########################

with tab2:
    try:
        st.subheader(defect_detection_subheader_title)
        
        img = image_select(
            label="Select image:",
            images=defect_images.split('\n'),
            captions=["Insulator #1", "Insulator #2", "Insulator #3", "Insulator #4"]
        )

        if st.button("Run Defect Detection"):
            st.divider()
            
            model_url = "https://clarifai.com/clarifai/insulator-defect-detection/models/insulator-condition-inception"
            
            with st.spinner("Processing defect detection..."):
                model = Model(url=model_url, pat=PAT)
                res_pmo = model.predict_by_url(img, input_type="image")
                outputs = res_pmo.outputs[0]
                regions = outputs.data.regions

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write("Predicted Defects")
                    image = Image.open(urllib.request.urlopen(img))
                    width, height = image.size
                    line_passes = 3

                    threshold = insulator_defect_threshold

                    concept_data = []
                    annotation_data = []

                    for region in regions:
                        top_row = round(region.region_info.bounding_box.top_row, 3)
                        left_col = round(region.region_info.bounding_box.left_col, 3)
                        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
                        right_col = round(region.region_info.bounding_box.right_col, 3)

                        for concept in region.data.concepts:
                            if concept.value >= threshold:
                                concept_data.append({
                                    "type": concept.name, 
                                    "confidence": concept.value,
                                    "width": width,
                                    "height": height,
                                    "top_row": top_row,
                                    "left_col": left_col,
                                    "bottom_row": bottom_row,
                                    "right_col": right_col
                                })
                                annotation_data.append(
                                    (f'{concept.name}', f'{concept.value:.3f}', tag_bg_color_1, tag_text_color_1)
                                )
                        
                        img1 = ImageDraw.Draw(image)

                        for concept in concept_data:
                            h1 = concept["top_row"] * height
                            w1 = concept["left_col"] * width
                            h2 = concept["bottom_row"] * height
                            w2 = concept["right_col"] * width

                            img1.rectangle((w1, h1, w2, h2), width=box_thickness, outline=box_color)

                            fontsize = int(image.height / 40)
                            font = ImageFont.load_default()

                            offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
                            for offset_x, offset_y in offsets:
                                img1.text(
                                    (w1 + offset_x + line_passes, h1 + offset_y + line_passes),
                                    concept["type"],
                                    align="left",
                                    fill='black',
                                    font=font
                                )
                            img1.text(
                                (w1 + line_passes, h1 + line_passes),
                                concept["type"],
                                align="left",
                                fill='white',
                                font=font
                            )

                    st.image(image)
                    
                    list_with_empty_strings = []
                    for item in annotation_data:
                        list_with_empty_strings.append(item)
                        list_with_empty_strings.append(" ")
                    
                    if list_with_empty_strings and list_with_empty_strings[-1] == " ":
                        list_with_empty_strings.pop()
                    
                    st.write("Detected Regions and Confidence Levels:")
                    annotated_text(*tuple(list_with_empty_strings))
        with st.expander("Details"):
          st.markdown("""
                      ### About Defect Detection for Insulators
                      Insulator Defect Detection detects and localizes specific defects in electrical insulators. It identifies any broken parts or flash-over damage and marks their locations on the image with bounding boxes.
                      
                      ### How This Works
                      1. **Image Selection**: 
                          - Choose one of the insulator images from the carousel
                          - Each image shows different types of potential defects
                      
                      2. **Output**:
                          - Click "Run Defect Detection" to analyze the selected image
                          - You'll see two views:
                              * Original Image: The unprocessed insulator image
                              * Predicted Defects: Shows bounding boxes around detected defects with:
                                  - Labels indicating the type of defect detected either "Broken Part" or "Flash Over"
                                  - Confidence scores for each detection
                      
                      ### Original Model
                      This implementation uses Clarifai's Insulator Defect Detection model.
                      - View the App here: [Clarifai Insulator Defect Detection](https://clarifai.com/clarifai/insulator-defect-detection)
                      """)
          project_details = st.text_area(
                                        "Add Your Notes:",
                                        height=100,
                                        key="defect_detection_notes",
                                        placeholder="Add any additional notes about insulator defect detection..."
                                        )
          if project_details:
                st.markdown("### Your Notes:")
                st.write(project_details)
    except Exception as e:
        st.error(f"Error in Defect Detection tab: {str(e)}")

######################
#### Crack Segmentation ####
######################

with tab3:
    try:
        st.subheader(segmentation_subheader_title)
        
        img = image_select(
            label="Select the image:",
            images=crack_images.split('\n'),
            captions=["Crack #1", "Crack #2", "Crack #3", "Crack #4"]
        )

        if st.button("Run Crack Segmentation"):
            st.divider()
            
            model_url = "https://clarifai.com/clarifai/crack-segmentation/models/crack-segmentation"
            
            with st.spinner("Processing crack segmentation..."):
                model = Model(url=model_url, pat=PAT)
                res_pmo = model.predict_by_url(img, input_type="image")

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write('Segmented Image')
                    display_segmented_image(res_pmo, img)
        
        with st.expander("Details"):
          st.markdown("""
                        ### About Crack Segmentation
                        Crack segmentation identifies and highlights cracks in surfaces using the segmentation model.
                        
                        ### How This Works
                        1. **Image Selection**: 
                            - Choose one of the images from the carousel
                            - Each image shows different types of surface cracks
                        
                        2. **Output**:
                            - Click "Run Crack Segmentation" to analyze the selected image
                            - You'll see two views:
                                * Original Image: The unprocessed surface image
                                * Segmented Image: Shows the detected cracks highlighted in green color (only appears if cracks are detected, returns a "No crack found" message if no cracks are detected)
                            - Below the segmented image, you'll see "crack" label with percentage of the crack detected
                               
                        ### Original Model
                        This implementation uses Clarifai's Crack Segmentation model.
                        - View the App here: [Clarifai Crack Segmentation](https://clarifai.com/clarifai/crack-segmentation)
                        """)
    
          project_details = st.text_area(
                "Add Your Notes:",
                height=100,
                key="crack_segmentation_notes",
                placeholder="Add any additional notes about crack detection and segmentation..."
            )
          if project_details:
              st.markdown("### Your Notes:")
              st.write(project_details)
    
    except Exception as e:
        st.error(f"Error in Crack Segmentation tab: {str(e)}")

#########################
#### Surface Defect Classification ####
#########################

with tab4:
    try:
        st.subheader(surface_defect_detection_subheader_title)
        
        # Add threshold slider
        threshold = surface_defect_threshold
        
        img = image_select(
            label="Select image:",
            images=surface_images.split('\n'),
            captions=["Surface #1", "Surface #2", "Surface #3", "Surface #4"]
        )
        
        if st.button("Run Surface Defect Classification"):
            st.divider()
            model_url = "https://clarifai.com/clarifai/surface-defects-sheet-metal/models/surface-defects"
            
            with st.spinner("Processing surface defect classification..."):
                model = Model(url=model_url, pat=PAT)
                surface_class_pred = model.predict_by_url(img, input_type="image")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)
                
                with col2:
                    st.write('Surface Defect Classification Results')
                    # Filter concepts based on threshold
                    filtered_concepts = [
                        x for x in surface_class_pred.outputs[0].data.concepts 
                        if x.value >= threshold
                    ]
                    
                    if not filtered_concepts:
                        st.info(f"No defects detected above the confidence threshold of {threshold:.2f}")
                    else:
                        concept_data = tuple([
                            (f'{x.name}', f'{x.value:.3f}', tag_bg_color_2, tag_text_color_2)
                            for x in filtered_concepts
                        ])
                        
                        # Add spacing between items
                        list_with_empty_strings = []
                        for item in concept_data:
                            list_with_empty_strings.append(item)
                            list_with_empty_strings.append(" ")
                        
                        if list_with_empty_strings and list_with_empty_strings[-1] == " ":
                            list_with_empty_strings.pop()
                        
                        concept_data = tuple(list_with_empty_strings)
                        annotated_text(*concept_data)

        with st.expander("Details"):
          st.markdown("""
                        ### About Surface Defect Classification
                        Surface Defect Classification identifies various types of defects on surface materials like sheet metal. It classifies different types of surface defects like crazing, inclusion, patches, pitted-surface, rolled-in-scale, scratches and provides confidence scores for each detected defect.
                        
                        ### How This Works
                        1. **Image Selection**: 
                            - Choose one of the surface images from the carousel
                            - Each image shows different types of surface conditions
                        
                        2. **Output**:
                            - Click "Run Surface Defect Classification" to analyze the selected image
                            - You'll see two views:
                                * Original Image: The unprocessed surface photo
                                * Classification Results: Shows detected defect types with:
                                    - Defect classification labels: crazing, inclusion, patches, pitted-surface, rolled-in-scale, scratches
                                    - Confidence scores for each detected defect
                        
                        ### Original Model
                        This implementation uses Clarifai's Surface Defect Classification model.
                        - View the App here: [Clarifai Surface Defect Classification](https://clarifai.com/clarifai/surface-defects-sheet-metal)
                        """)
                        
          project_details = st.text_area(
              "Add Your Notes:",
              height=100,
              key="surface_defect_notes",
              placeholder="Add any additional notes about surface defect classification..."
          )
          if project_details:
              st.markdown("### Your Notes:")
              st.write(project_details)

    except Exception as e:
        st.error(f"Error in Surface Defect Classification tab: {str(e)}")

####################
####  FOOTER    ####
####################

footer(st)