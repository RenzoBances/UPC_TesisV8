import pandas as pd
import numpy as np

# Utilitarios  functions
def create_df_results():
    df_results = pd.DataFrame({
        'id_exercise'               : pd.Series(dtype='str'),   #1
        'DateTime_Start'            : pd.Series(dtype='str'),   #2
        'n_poses'                   : pd.Series(dtype='int'),   #3
        'n_sets'                    : pd.Series(dtype='int'),   #4
        'n_reps'                    : pd.Series(dtype='int'),   #5
        'total_poses'               : pd.Series(dtype='int'),   #6
        'seconds_rest_time'         : pd.Series(dtype='int'),   #7
        'Class'                     : pd.Series(dtype='str'),   #8
        'Prob'                      : pd.Series(dtype='float'), #9
        'count_pose_g'              : pd.Series(dtype='int'),   #10
        'count_pose'                : pd.Series(dtype='int'),   #11
        'count_rep'                 : pd.Series(dtype='int'),   #12
        'count_set'                 : pd.Series(dtype='int'),   #13
        #push_up
        'right_elbow_angles_pu'     : pd.Series(dtype='float'), #14
        'right_hit_angles_pu'       : pd.Series(dtype='float'), #15
        'right_knee_angles_pu'      : pd.Series(dtype='float'), #16
        #curl_up
        'right_shoulder_angles_cu'  : pd.Series(dtype='float'), #17
        'right_hit_angles_cu'       : pd.Series(dtype='float'), #18
        'right_knee_angles_cu'      : pd.Series(dtype='float'), #19
        #front_plank
        'right_shoulder_angles_fp'  : pd.Series(dtype='float'), #20
        'right_hit_angles_fp'       : pd.Series(dtype='float'), #21
        'right_ankle_angles_fp'     : pd.Series(dtype='float'), #22
        #forward_lunge
        'right_hit_angles_fl'       : pd.Series(dtype='float'), #23
        'right_knee_angles_fl'      : pd.Series(dtype='float'), #24
        'left_knee_angles_fl'       : pd.Series(dtype='float'), #25
        #bird_dog
        'right_shoulder_angles_bd'  : pd.Series(dtype='float'), #26
        'right_hit_angles_bd'       : pd.Series(dtype='float'), #27
        'right_knee_angles_bd'      : pd.Series(dtype='float'), #28
        'left_knee_angles_bd'       : pd.Series(dtype='float'), #29
        'right_elbow_angles_bd'     : pd.Series(dtype='float'), #30
        'left_elbow_angles_bd'      : pd.Series(dtype='float'), #31
        })
    return df_results


def add_row_df_results(df_results,
                       id_exercise,             #1
                       DateTime_Start,          #2
                       n_poses,                 #3
                       n_sets,                  #4
                       n_reps,                  #5
                       total_poses,             #6
                       seconds_rest_time,       #7
                       Class,                   #8
                       Prob,                    #9
                       count_pose_g,            #10
                       count_pose,              #11
                       count_rep,               #12
                       count_set,               #13
                       #push_up
                       right_elbow_angles_pu,   #14
                       right_hit_angles_pu,     #15
                       right_knee_angles_pu,    #16
                       #curl_up
                       right_shoulder_angles_cu,#17
                       right_hit_angles_cu,     #18
                       right_knee_angles_cu,    #19
                       #front_plank
                       right_shoulder_angles_fp,#20
                       right_hit_angles_fp,     #21
                       right_ankle_angles_fp,   #22
                       #forward_lunge 
                       right_hit_angles_fl,     #23
                       right_knee_angles_fl,    #24
                       left_knee_angles_fl,     #25
                       #bird_dog
                       right_shoulder_angles_bd,#26
                       right_hit_angles_bd,     #27
                       right_knee_angles_bd,    #28
                       left_knee_angles_bd,     #29
                       right_elbow_angles_bd,   #30
                       left_elbow_angles_bd,    #31
                       ):
    
    df_results.loc[len(df_results.index)] = [
        id_exercise,        #1 - str - id_exercise
        DateTime_Start,     #2 - str - DateTime_Start
        n_poses,            #3 - int - n_poses
        n_sets,             #4 - int - n_sets
        n_reps,             #5 - int - n_reps
        total_poses,        #6 - int - total_poses
        seconds_rest_time,  #7 - int - seconds_rest_time
        Class,              #8 - str - Class
        Prob,               #9 - float - Prob
        count_pose_g,       #10 - int - count_pose_g
        count_pose,         #11 - int - count_pose_
        count_rep,          #12 - int - count_rep
        count_set,          #13 - int - count_set
        #push_up
        right_elbow_angles_pu,   #14 - float
        right_hit_angles_pu,     #15 - float
        right_knee_angles_pu,    #16 - float
        #curl_up
        right_shoulder_angles_cu,#17 - float
        right_hit_angles_cu,     #18 - float
        right_knee_angles_cu,    #19 - float
        #front_plank
        right_shoulder_angles_fp,#20 - float
        right_hit_angles_fp,     #21 - float
        right_ankle_angles_fp,   #22 - float
        #forward_lunge 
        right_hit_angles_fl,     #23 - float
        right_knee_angles_fl,    #24 - float
        left_knee_angles_fl,     #25 - float
        #bird_dog
        right_shoulder_angles_bd,#26 - float
        right_hit_angles_bd,     #27 - float
        right_knee_angles_bd,    #28 - float
        left_knee_angles_bd,     #29 - float
        right_elbow_angles_bd,   #30 - float
        left_elbow_angles_bd,    #31 - float
        ]
    return df_results