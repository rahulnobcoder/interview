import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from functions.models import models_dict
from functions.helper import extract_faces_from_frames,make_pdf
from functions.video import eyebrow,detect_blinks,detect_yawns,detect_smiles
from functions.valence_arousal import va_predict
from functions.fer import fer_predict,plot_graph
from functions.helper import plot_facial_expression_graphs
from moviepy.editor import VideoFileClip
import json 
# from trash import detect_eyes_in_faces
import pandas as pd 
from typing import Callable
from functions.audio import extract_audio_features
asrmodel=models_dict['asrmodel']
asrproc=models_dict['asrproc']
sentipipe=models_dict['sentipipe']
valence_arousal_model=models_dict['valence_fer'][1]
val_ar_feat_model=models_dict['valence_fer'][0]
fer_model=models_dict['fer']
smile_cascade=models_dict['smile_cascade']
dnn_net=models_dict['face'][0]
predictor=models_dict['face'][1]
fps=30
session_data={}

def analyze_live_video(video_path: str, uid: str, user_id: str, count: int, final: bool, log: Callable[[str], None]):
    try:
        global session_data
        if uid not in session_data:
            session_data[uid]={
                "vcount":[],
				"duration":[],
                    
                "audio":[],

                "blinks":[],
                "yawn":[],
                "smile":[],
                "eyebrow":[],

                "fer": [],
				"valence":[],
				"arousal":[],
				"stress":[],
            }
        print(f"UID: {uid}, User ID: {user_id}, Count: {count}, Final: {final}, Video: {video_path}")
        print(f"analysing video for question - {count}")

        output_dir = os.path.join('output', uid)
        os.makedirs(output_dir,exist_ok=True)
        
        folder_path=os.path.join(output_dir,f'{count}')
        os.makedirs(folder_path,exist_ok=True)
        meta_data_path=os.path.join(folder_path,'metadata.json')
        valence_plot=os.path.join(folder_path,"vas.png")
        word_cloud=os.path.join(folder_path,'wordcloud.jpg')
        df_path=os.path.join(folder_path,'data.csv')
        pdf_filename = os.path.join(folder_path,"formatted_output_with_plots.pdf")

        video_clip=VideoFileClip(video_path)
        video_clip=video_clip.set_fps(fps)
        duration=video_clip.duration
        print(duration)
        audio=video_clip.audio
        audio_path = os.path.join(folder_path,'extracted_audio.wav')
        print(audio_path)
        audio.write_audiofile(audio_path)
        video_frames=[frame for frame in video_clip.iter_frames()]
        faces, landmarks, sizes=extract_faces_from_frames(video_frames,dnn_net,predictor)


        # faces=[extract_face(frame) for frame in tqdm(video_frames)]
        af,pitches=extract_audio_features(audio_path,asrmodel,asrproc,sentipipe,duration,word_cloud)
        pitches=[float(pitch) for pitch in pitches]

        fer_emotions,class_wise_frame_count,em_tensors=fer_predict(faces,fps,fer_model)
        valence_list,arousal_list,stress_list=va_predict(valence_arousal_model,val_ar_feat_model,faces,list(em_tensors))
        timestamps=[j/fps for j in range(len(valence_list))]

        eyebrow_dist=eyebrow(landmarks,sizes)
        print('eyebrow done')

        blink_count, ear_ratios=detect_blinks(landmarks,sizes,fps)
        ear_ratios=[float(pitch) for pitch in ear_ratios]
        print('blinks done',blink_count)
        smiles, smile_ratios, total_smiles, smile_durations,smile_threshold=detect_smiles(landmarks,sizes)
        smile_ratios=[float(smile) for smile in smile_ratios]
        print('smiles done',total_smiles)
        yawns, yawn_ratios, total_yawns, yawn_durations=detect_yawns(landmarks,sizes)
        print('ywan done')

        thresholds=[smile_threshold,0.225,0.22]
        buffer = plot_facial_expression_graphs(smile_ratios, ear_ratios, yawn_ratios, thresholds, 'path_to_save_plot.pdf')

        # print("detect_eyes : ",detect_eyes_in_faces(faces))

        y_vals = [valence_list, arousal_list, stress_list,eyebrow_dist,pitches]
        labels = ['Valence', 'Arousal', 'Stress',"EyeBrowDistance","Pitch"]
        buf=plot_graph(timestamps, y_vals, labels, valence_plot)
        print('graph_plotted')
        meta_data={}
        meta_data['duration']=duration
        meta_data['facial_emotion_recognition'] = {
			"class_wise_frame_count": class_wise_frame_count,
		}
        meta_data['audio']=af


        make_pdf(pdf_filename,meta_data,buf,buffer)

        with open(meta_data_path, 'w') as json_file:
            json.dump(meta_data, json_file, indent=4)
        df=pd.DataFrame(
            {
                'timestamps':timestamps,
                'fer': fer_emotions,
                'valence': valence_list,
                'arousal': arousal_list,
                'stress': stress_list,
                'eyebrow':eyebrow_dist,
            }
        )
        df.to_csv(df_path,index=False)
    except Exception as e:
            print("Error analyzing video...: ", e)
            
# analyze_live_video('s1.mp4','1',1,1,True,print)

