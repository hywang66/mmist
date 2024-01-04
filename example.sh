
# model settings
stylegan3_path="./stylegan3" # StyleGAN3 repo path
stylegan3_pkl="./stylegan3/models/wikiart-1024-stylegan3-t-17.2Mimg.pkl" # StyleGAN3 wikiart pre-trained model path
adaattn_path="./AdaAttN" # AdaAttN repo path

# experiment settings
exp_name="exp_mmist" # experiment name, which determines the output folder name

# set up the content images
# content_paths="./contents/avril.jpg ./contents/cornell.jpg ./contents/modern.jpg" # use multiple content images
content_paths="./contents" # or use a folder / multiple folders of content images

# set up the multimodal style inputs
sty_text="fire" # the style text description
sty_img="./styles/mondrian.jpg" # the source style image


# Fisrt, generate the style representations
# alpha_text are the weights for the style text
# alpha_img are the weights for the style image
MKL_THREADING_LAYER=GNU python gen_style_reps.py \
    --exp_name $exp_name \
    --sty_text "$sty_text" \
    --sty_img $sty_img \
    --alpha_text 500 \
    --alpha_img 500 \
    --stylegan3_path $stylegan3_path \
    --stylegan3_pkl $stylegan3_pkl 

# Then, apply the style representations to the content images to generate stylized images
python apply_style_reps.py \
    --style_reps_dir outputs/$exp_name/style_reps \
    --output_dir outputs/$exp_name/stylized_imgs \
    --content_paths $content_paths \
    --adaattn_path $adaattn_path 

