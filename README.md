# darpa_code


Codes
Download SCUNet models
python main_download_pretrained_models.py --models "SCUNet" --model_dir "model_zoo"

For the Radiograph denoising

python polished_main_test_scunet_gray_darpa.py


For 3D reconstrcuction 
python dip_reconstruction_refactored.py --file_path=/egr/research-slim/shared/TomographyData256/volume_0.pt --save_path=/egr/research-slim/liangs16/deep-image-prior-master_8x/ct_result//volume_full_recon_LowBlur_SCU_blend_TV.pt --device=cuda:2 --I_0=100 --blur_size=7 --blur_sigma=3 --bm3d_sigma=4 --z_consistency_lambda=1 --tv_lambda=1e-6
