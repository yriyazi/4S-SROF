import  PIL, cv2, os, processor.visualization as visualization, utils, model, tqdm
import  numpy                   as      np
import  pandas                  as      pd
from    scipy.signal            import  savgol_filter
from    .baseline_detection     import  Baseline
from    .criteria_definition    import  receding_pixel_selection, advancing_pixel_selection,poly_fitting, right_angle, left_angle

def main_processor(ad,
                   framework,
                   num_px_ratio,
                   cm_on_pixel_ratio,
                   fps,
                   error_handling_kernel_size):
    angle, rotated1, red1_xs, red1_ys, red2_xs, red2_ys =   utils.slope_measurement(ad)
    baseline                                            =   min(utils.find_reds(rotated1)[1])-1
    name_files                                          =   utils.load_files(ad)
    img_frame                                           =   cv2.imread(os.path.join(ad,name_files[0]))
    img_frame_rotated                                   =   utils.rotate_image(img_frame, angle)

    adv_list, rec_list, contact_line_length_list, x_center_list, y_center_list, middle_angle_degree_list,processed_number_list=[],[],[],[],[],[],[]
    rec_angle_point_list, adv_angle_point_list=[],[]

    for file_number in tqdm.tqdm(range(1, len(name_files))):
        img_drop=cv2.imread(os.path.join(ad,name_files[file_number]))
        img_drop_rotated=utils.rotate_image(img_drop, angle)

        #drop diff
        diff_img=cv2.absdiff(img_drop_rotated, img_frame_rotated)

        #drop cropping
        BaseL=Baseline(baseline,
                       middle_drop_height           =25,
                       drop_start_height            =3*3,
                       object_detection_threshold   =40)
        
        drop_reflection,*dim=BaseL.drop_cropping(diff_img, 
                                                 x_left_margin=30,
                                                 x_right_margin=60,
                                                 y_up_margin=10)
        
        just_drop =diff_img[dim[2]:baseline,dim[0]:dim[1],:]

        #super resolution    
        upscaled_image=model.upscale_image(cv2.cvtColor(just_drop.astype('uint8'), cv2.COLOR_BGR2RGB),framework)

        #utilizing morphological transformation to remove noises
        kernel          = np.ones(error_handling_kernel_size,np. uint8) 
        upscaled_image  = cv2.morphologyEx(np.array(upscaled_image),
                                           cv2.MORPH_CLOSE, kernel)

        #keeping just external pixels as droplet curvature
        i_list, j_list =model.edge_extraction( upscaled_image, thr=40)

        #extracting the desired number of pixels as input of the polynomial fitting 
        left_number_of_pixels=int(150*num_px_ratio)
        right_number_of_pixels=int(65*num_px_ratio)
        i_left, j_left=advancing_pixel_selection(i_list,j_list,
                                                 left_number_of_pixels=left_number_of_pixels)
        i_right, j_right=receding_pixel_selection(i_list,j_list,
                                                  right_number_of_pixels=right_number_of_pixels)

        #rotation for fitting, it can increase the accuracy to rotate 90 degrees then fit the polynomial
        i_left_rotated, j_left_rotated      =j_left,    i_left       
        i_right_rotated,j_right_rotated     =j_right,   i_right   

        left_polynomial_degree  =   3
        right_polynomial_degree =   2
        i_poly_left_rotated,    j_poly_left_rotated =poly_fitting(i_left_rotated,j_left_rotated,polynomial_degree=left_polynomial_degree,line_space=left_number_of_pixels)
        i_poly_right_rotated,   j_poly_right_rotated=poly_fitting(i_right_rotated,j_right_rotated,polynomial_degree=right_polynomial_degree,line_space=right_number_of_pixels)

        right_angle_degree, right_angle_point       =right_angle(i_poly_right_rotated, j_poly_right_rotated,1)
        left_angle_degree,  left_angle_point        =left_angle (i_poly_left_rotated, j_poly_left_rotated,1)
        

        left_polynomial_degree ,left_number_of_pixels   = utils.angle_polynomial_order(num_px_ratio,left_polynomial_degree)
        right_polynomial_degree ,right_number_of_pixels = utils.angle_polynomial_order(num_px_ratio,left_polynomial_degree)

        #9. extracting the desired number of pixels as input of the polynomial fitting 
        i_left, j_left=advancing_pixel_selection(i_list,j_list, left_number_of_pixels=left_number_of_pixels)
        i_right, j_right=receding_pixel_selection(i_list,j_list, right_number_of_pixels=right_number_of_pixels)

        #10. rotation for fitting, it can increase the accuracy to rotate 90 degrees and then fit the polynomial
        i_left_rotated,j_left_rotated=j_left,i_left       
        i_right_rotated,j_right_rotated=j_right,i_right   
        i_poly_left_rotated, j_poly_left_rotated    = poly_fitting(i_left_rotated,j_left_rotated,polynomial_degree=left_polynomial_degree,line_space=left_number_of_pixels)
        i_poly_right_rotated, j_poly_right_rotated  = poly_fitting(i_right_rotated,j_right_rotated,polynomial_degree=right_polynomial_degree,line_space=right_number_of_pixels)
        j_poly_left=i_poly_left_rotated
        i_poly_left=j_poly_left_rotated
        j_poly_right=i_poly_right_rotated
        i_poly_right=j_poly_right_rotated
        x_cropped=dim[0]

        distance = (x_cropped) * 3
        
        address=os.path.join(ad,'SR_edge',name_files[file_number])
        adv, rec,rec_angle_point, adv_angle_point, contact_line_length, x_center, y_center, middle_angle_degree=visualization.visualize(address, 
                                                                                                                                    distance+np.array(i_list),j_list,distance+np.array(i_left),j_left,distance+np.array(i_right),j_right,
                                                                                                                                    j_poly_left,distance+np.array(i_poly_left),j_poly_right,distance+np.array(i_poly_right),x_cropped,
                                                                                                                                    distance+np.array(i_poly_left_rotated), j_poly_left_rotated, distance+np.array(i_poly_right_rotated),
                                                                                                                                    j_poly_right_rotated, cm_on_pixel=cm_on_pixel_ratio, middle_line_switch=1)
        processed_number_list.append(int(name_files[file_number].split(".")[0].split("S0001")[-1]))
        adv_list.append(adv)
        rec_list.append(rec)
        adv_angle_point_list.append(adv_angle_point)
        rec_angle_point_list.append(rec_angle_point)
        contact_line_length_list.append(contact_line_length)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        middle_angle_degree_list.append(middle_angle_degree)

    vel=[]

    for i in range(len(x_center_list)-1):
        vel=vel+[x_center_list[i+1]-x_center_list[i]]

    vel=np.array(vel)*fps

    df=pd.DataFrame([processed_number_list,
                    np.arange(0, 1/fps*len(vel), 1/fps),
                    x_center_list,
                    adv_list,
                    rec_list,
                    contact_line_length_list,
                    y_center_list,
                    middle_angle_degree_list,
                    vel]).T
    df=df[:-1]

    df.columns=['file number',
                "time (s)",
                'x_center (cm)',
                'adv (degree)',
                'rec (degree)',
                'contact_line_length (cm)',
                'y_center (cm)',
                'middle_angle_degree (degree)',
                'velocity (cm/s)']

    window_length   = 9
    polyorder       = 2

    df["adv (degree)"]                      = savgol_filter(df["adv (degree)"],                 window_length, polyorder)
    df["rec (degree)"]                      = savgol_filter(df["rec (degree)"],                 window_length, polyorder)
    df["contact_line_length (cm)"]          = savgol_filter(df["contact_line_length (cm)"],     window_length, polyorder)
    df["y_center (cm)"]                     = savgol_filter(df["y_center (cm)"],                window_length, polyorder)
    df["middle_angle_degree (degree)"]      = savgol_filter(df["middle_angle_degree (degree)"], window_length, polyorder)
    df["velocity (cm/s)"]                   = savgol_filter(df["velocity (cm/s)"],              window_length, polyorder)

    df.to_csv(os.path.join(ad,'SR_result','result.xlsx'), index=False)

    return df